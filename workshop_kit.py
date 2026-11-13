"""Shared helpers for the "Govern Your Agents" workshop.

The notebook and the tests both import this module, so there is one
definition of the agent's prompt, tools, model paths and reference loop.

The constants in the first section determine every node ID shown on the
slides. Change one of them and the IDs change, which is the point of a
content-addressed execution tree, but the slides will no longer match.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

try:
    from pollard import (
        ActionSpec,
        ApprovalPolicy,
        Budget,
        BudgetExceeded,
        IntegrityError,
        MissingRecording,
        PolicyViolation,
        Registry,
        Runtime,
        SQLiteStore,
        decide_approval,
        request_approval,
        seal,
        verify,
    )
    from pollard.approvals import ApprovalRequest
except ImportError as exc:  # durable approvals arrived in Pollard 1.6.1
    raise ImportError(f"this workshop needs pollard 1.6.1 or newer ({exc})") from exc

# --------------------------------------------------------------------------
# Constants that decide the node IDs
# --------------------------------------------------------------------------
DB = "workshop.db"
RUN_LABEL = "refund-agent"
SCRIPTED_MODEL = "scripted-model"
TASK = "Order A-1001 arrived damaged. Please refund it."
SYSTEM = (
    "You are a support agent for an online store. "
    "Reply with exactly one JSON object and nothing else.\n"
    'To call a tool: {"tool": "<name>", "args": {...}}\n'
    'When you are done: {"final": "<one sentence for the customer>"}\n'
    "Tools:\n"
    "- lookup_order(order_id: string) reads one order and returns its status and paid_cents.\n"
    "- issue_refund(order_id: string, cents: integer) returns money to the customer; cents is 1 to 10000.\n"
    "Rules: call one tool per reply. Look up the order before any refund. "
    "If the order was delivered and the customer reports damage, refund the full paid_cents."
)
TOOL_RESULT_PREFIX = "Tool result for "

# Which fields of a tool result the printed line shows.
SHOW = {"lookup_order": ("status",), "issue_refund": ("refund_id", "cents")}

# Live call counters: a handler or model function that really runs adds one.
LIVE = {"model": 0, "lookup_order": 0, "issue_refund": 0}


def reset_counters() -> None:
    for key in LIVE:
        LIVE[key] = 0


def live_calls() -> str:
    """The counters as the slides print them, for example 'model 1, lookup_order 1'."""
    return ", ".join(f"{name} {count}" for name, count in LIVE.items() if count) or "none"


# --------------------------------------------------------------------------
# Messages: plain chat roles, so the same payload works on any chat endpoint
# --------------------------------------------------------------------------
def first_messages(task: str = TASK) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]


def assistant_message(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def tool_message(name: str, result: dict[str, Any]) -> dict[str, str]:
    """A tool result goes back to the model as a user message, without the usage block."""
    body = {key: value for key, value in result.items() if key != "usage"}
    return {"role": "user", "content": f"{TOOL_RESULT_PREFIX}{name}: {json.dumps(body, sort_keys=True)}"}


def parse_decision(text: str) -> dict[str, Any]:
    """Read the model's reply: either {"tool": name, "args": {...}} or {"final": text}.

    Text around the JSON object (a code fence, a sentence) is ignored. Anything else raises ValueError.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the model reply holds no JSON object")
    try:
        decision = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the model reply is not valid JSON ({exc.msg})") from exc
    if isinstance(decision, dict) and isinstance(decision.get("final"), str):
        return {"final": decision["final"]}
    if isinstance(decision, dict) and isinstance(decision.get("tool"), str) and isinstance(decision.get("args"), dict):
        return {"tool": decision["tool"], "args": decision["args"]}
    raise ValueError("the model reply is neither a tool call nor a final answer")


def describe(name: str, result: dict[str, Any]) -> str:
    """One printed line for a tool result, for example 'refund_id R-77, cents 4200'."""
    if name in SHOW:
        return ", ".join(f"{key} {result[key]}" for key in SHOW[name] if key in result)
    return json.dumps({key: value for key, value in result.items() if key != "usage"}, sort_keys=True)


# --------------------------------------------------------------------------
# Tools and registry
# --------------------------------------------------------------------------
def sentinel(_payload: Any) -> Any:
    """Stands in for every model and tool during strict replay. It must never run."""
    raise AssertionError("strict replay must never call a model or a tool")


def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
    LIVE["lookup_order"] += 1
    return {"order_id": args["order_id"], "status": "delivered", "paid_cents": 4200,
            "usage": {"input_tokens": 0, "output_tokens": 0}}


def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
    LIVE["issue_refund"] += 1
    return {"refund_id": "R-77", "cents": args["cents"], "usage": {"input_tokens": 0, "output_tokens": 0}}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def make_registry(replay: bool = False) -> Registry:
    """Two registered actions. With replay=True every handler is the sentinel."""
    return Registry([
        ActionSpec("lookup_order", "1", "Read one order",
                   _object({"order_id": {"type": "string"}}, ["order_id"]),
                   False, sentinel if replay else lookup_order),
        ActionSpec("issue_refund", "1", "Return money to a customer",
                   _object({"order_id": {"type": "string"},
                            "cents": {"type": "integer", "minimum": 1, "maximum": 10000}}, ["order_id", "cents"]),
                   True, sentinel if replay else issue_refund),
    ])


registry = make_registry()


def has_side_effects(name: str, reg: Registry | None = None) -> bool:
    """True for a registered action that changes the world. Unknown names are left to the registry to refuse."""
    try:
        return bool((reg or registry).get(name).side_effects)
    except KeyError:
        return False


# --------------------------------------------------------------------------
# Model paths: scripted (always available), endpoint, colab
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPath:
    path: str                                   # "scripted", "endpoint", "colab" or "recording"
    name: str                                   # the model name that goes into the identity payload
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    note: str


def _tool_results(messages: list[dict[str, str]]) -> int:
    return sum(1 for m in messages if m["role"] == "user" and m["content"].startswith(TOOL_RESULT_PREFIX))


def _estimated_usage(messages: list[dict[str, str]], text: str) -> dict[str, int]:
    """A character-count estimate, used only where no provider usage exists."""
    return {"input_tokens": sum(len(m["content"]) for m in messages) // 4, "output_tokens": len(text) // 4}


def scripted_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for a model: one decision per completed tool call."""
    LIVE["model"] += 1
    decision = [
        {"tool": "lookup_order", "args": {"order_id": "A-1001"}},
        {"tool": "issue_refund", "args": {"order_id": "A-1001", "cents": 4200}},
        {"final": "Refund of $42.00 issued for order A-1001."},
    ][min(_tool_results(payload["messages"]), 2)]
    text = json.dumps(decision, sort_keys=True)
    return {"text": text, "usage": _estimated_usage(payload["messages"], text), "usage_estimated": True}


SCRIPTED = ModelPath("scripted", SCRIPTED_MODEL, scripted_model, "deterministic and offline")

_ENDPOINT_SETTINGS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "WORKSHOP_MODEL")


def _load_env_file(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a local .env file. Values already in the environment win."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return
    for line in lines:
        key, separator, value = line.strip().partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if separator and key and value and not key.startswith("#") and key not in os.environ:
            os.environ[key] = value


def _load_colab_secrets() -> None:
    """In Colab, copy the endpoint settings from Colab Secrets into the environment, if they are there."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return
    for key in _ENDPOINT_SETTINGS:
        if not os.environ.get(key):
            with contextlib.suppress(Exception):
                value = userdata.get(key)
                if value:
                    os.environ[key] = value


def _endpoint_path(model_name: str | None) -> ModelPath:
    """Any OpenAI-compatible endpoint, through Pollard's chat-completions adapter. Usage is the provider's."""
    from openai import OpenAI  # type: ignore[import-not-found]
    from pollard.adapters.openai import make_chat_completions_fn

    _load_colab_secrets()
    _load_env_file()
    name = model_name or os.environ.get("WORKSHOP_MODEL")
    if not name:
        raise RuntimeError("set WORKSHOP_MODEL to a model name")
    client = OpenAI(timeout=30.0, max_retries=1)  # reads OPENAI_API_KEY and OPENAI_BASE_URL
    client.chat.completions.create(model=name, messages=[{"role": "user", "content": "Reply with OK."}])
    call = make_chat_completions_fn(client)

    def fn(payload: dict[str, Any]) -> dict[str, Any]:
        LIVE["model"] += 1
        return call(payload)

    return ModelPath("endpoint", name, fn, "provider usage, exact token counts")


def _colab_path(model_name: str | None) -> ModelPath:
    """Colab's built-in model library. It returns text only, so token counts are estimates."""
    from google.colab import ai  # type: ignore[import-not-found]

    name = model_name
    if not name:
        names = [str(getattr(item, "name", item)) for item in ai.list_models()]
        name = next((n for n in names if "flash" in n and "lite" not in n), names[0])
    ai.generate_text("Reply with OK.", model_name=name)

    def fn(payload: dict[str, Any]) -> dict[str, Any]:
        LIVE["model"] += 1
        turns = [f"{m['role'].upper()}: {m['content']}" for m in payload["messages"]]
        text = str(ai.generate_text("\n\n".join(turns + ["ASSISTANT:"]), model_name=payload["model"]))
        return {"text": text, "usage": _estimated_usage(payload["messages"], text), "usage_estimated": True}

    return ModelPath("colab", name, fn, "text only, estimated token counts")


_PATHS: dict[str, Callable[[str | None], ModelPath]] = {"endpoint": _endpoint_path, "colab": _colab_path}


def _without_key(text: str) -> str:
    """Keep a configured API key out of anything that gets printed."""
    key = os.environ.get("OPENAI_API_KEY") or ""
    return text.replace(key, "***") if len(key) >= 8 else text


def choose_model(path: str | None = None, *, model_name: str | None = None, quiet: bool = False) -> ModelPath:
    """Pick a model path. A real path that fails its one-call check falls back to the scripted path.

    With no argument the path comes from WORKSHOP_MODEL_PATH (environment or .env), else it is "scripted".
    """
    say = (lambda *a: None) if quiet else print
    if path is None:
        _load_env_file()
        path = os.environ.get("WORKSHOP_MODEL_PATH") or "scripted"
    if path != "scripted" and path not in _PATHS:
        raise ValueError("path must be 'scripted', 'endpoint' or 'colab'")
    chosen = SCRIPTED
    if path in _PATHS:
        try:
            chosen = _PATHS[path](model_name)
        except Exception as exc:  # any failure of a real path must leave the room with a working one
            say(f"Model path '{path}' is not available: {type(exc).__name__}: {_without_key(str(exc))[:120]}")
            say("Falling back to the scripted path.")
    say(f"Active model path: {chosen.path} ({chosen.name}; {chosen.note})")
    return chosen


def recorded_model(db: str) -> ModelPath:
    """The model name stored in a recording, so strict replay can ask for the recorded steps."""
    with SQLiteStore(db, read_only=True) as store:
        for root in store.roots():
            for node in store.walk(root):
                if node.kind == "model_call":
                    return ModelPath("recording", str(node.payload["model"]), sentinel, "served from a recording")
    raise ValueError(f"{db} holds no model call")


# --------------------------------------------------------------------------
# The reference agent loop: what the exercise cells build up to
# --------------------------------------------------------------------------
def agent(db: str = DB, *, steps: int, gate: bool, mode: str = "hybrid", task: str = TASK,
          model: ModelPath | None = None, label: str = RUN_LABEL, quiet: bool = False) -> dict[str, Any]:
    """Run the agent once from the root. Hybrid mode serves recorded steps and executes missing ones.

    Returns the IDs of the steps on the path, a status ("done", "budget", "policy", "missing",
    "protocol" or "rejected"), the refusal node, the approval request and the live call counts.
    Every branch under one label draws on the same budget, so give an experiment its own label.
    """
    say = (lambda *a: None) if quiet else print
    model = model or SCRIPTED
    reset_counters()
    out: dict[str, Any] = {"path": [], "status": None, "refusal": None, "request": None, "root": None}
    replay = mode == "replay"
    with SQLiteStore(db, read_only=replay) as store:
        reg = make_registry(replay)
        runtime = Runtime(store, registry=reg, policies=[ApprovalPolicy(store)] if gate else [], mode=mode)
        fn = sentinel if replay else model.fn
        try:
            with runtime.run(label, budget=Budget(steps=steps)) as run:
                out["root"] = run.root_id
                messages = first_messages(task)
                while True:
                    node = run.model_call({"model": model.name, "messages": messages}, fn=fn)
                    out["path"].append(node.id[:8])
                    text = node.result.get("text") or ""
                    try:
                        decision = parse_decision(text)
                    except ValueError as problem:
                        out["status"] = "protocol"
                        say(f"  STOPPED: {problem}")
                        say("  This reply is recorded, so a rerun serves it again. Use reset() or the scripted path.")
                        break
                    messages = messages + [assistant_message(text)]
                    if "final" in decision:
                        out["status"] = "done"
                        say("  FINAL:", decision["final"])
                        break
                    name, args = decision["tool"], decision["args"]
                    request = None
                    if gate and has_side_effects(name, reg):
                        try:
                            request = request_approval(run, name, args)
                            out["request"] = request.to_dict()
                        except (TypeError, ValueError):
                            pass  # arguments the registry will refuse: no approval to ask for
                    try:
                        result = run.tool_call(name, args)
                    except PolicyViolation as refusal:
                        out["status"], out["refusal"] = "policy", refusal.refusal_id[:8]
                        say(f"  REFUSED by policy: {refusal}")
                        say(f"  refusal node {refusal.refusal_id[:8]}"
                            + (f", request {request.id[:8]}" if request is not None else ""))
                        break
                    except TypeError as problem:  # for example a float in the arguments
                        out["status"] = "rejected"
                        say(f"  REJECTED before dispatch: {problem}")
                        break
                    out["path"].append(result.id[:8])
                    say(f"  tool {name} -> {describe(name, result.result)}")
                    messages = messages + [tool_message(name, result.result)]
        except BudgetExceeded as refusal:
            out["status"], out["refusal"] = "budget", refusal.refusal_id[:8]
            say(f"  REFUSED by budget: {refusal}")
            say(f"  refusal node {refusal.refusal_id[:8]}")
        except MissingRecording as miss:
            out["status"] = "missing"
            say("  MissingRecording:", str(miss)[:60])
    out["live"] = dict(LIVE)
    say("  live calls:", live_calls())
    return out


# --------------------------------------------------------------------------
# Approval, evidence and housekeeping
# --------------------------------------------------------------------------
def approve(request: dict[str, Any], reviewer: str, *, approved: bool = True, db: str = DB) -> str:
    """Record a reviewer's decision for one approval request. Returns the decision node ID prefix.

    The same decision twice is the same node. A different decision for the same request raises IntegrityError.
    """
    with SQLiteStore(db) as store:
        decision = decide_approval(store, ApprovalRequest.from_dict(request), approved=approved, reviewer=reviewer)
    return decision.id[:8]


def verify_tree(store: Any, root: str) -> bool:
    """verify() checks one node and its ancestors, so check every node, as `pollard verify` does."""
    return all(verify(store, node.id).ok for node in store.walk(root))


def evidence(root: str, db: str = DB) -> tuple[bool, str, int]:
    """Whole-tree verification, the seal digest and the node count."""
    with SQLiteStore(db, read_only=True) as store:
        return verify_tree(store, root), seal(store, root).digest, sum(1 for _ in store.walk(root))


def tamper(db: str = DB, copy: str = "tampered.db") -> str:
    """Copy the store and edit one recorded result in the copy. Returns what was edited, or "" if nothing was.

    The edit shrinks the recorded refund. A tree without a refund gets its first model reply edited instead.
    """
    if not os.path.exists(db):
        raise FileNotFoundError(db)
    edit = ""
    with contextlib.closing(sqlite3.connect(db)) as source, contextlib.closing(sqlite3.connect(copy)) as target:
        source.backup(target)
        rows = [(node_id, kind, json.loads(raw)) for node_id, kind, raw in target.execute(
            "select id, kind, result from nodes where result is not null order by rowid")]
        for node_id, kind, result in rows:
            if kind == "tool_call" and isinstance(result, dict) and "refund_id" in result \
                    and isinstance(result.get("cents"), int):
                old = result["cents"]
                result["cents"] = max(1, old // 100)
                edit = f"refund result {node_id[:8]}: cents {old} -> {result['cents']}"
                break
        else:
            for node_id, kind, result in rows:
                if kind == "model_call" and isinstance(result, dict) and isinstance(result.get("text"), str):
                    result["text"] += " "
                    edit = f"model reply {node_id[:8]}: one space added"
                    break
        if edit:
            target.execute("update nodes set result = ? where id = ?",
                           (json.dumps(result, sort_keys=True, separators=(",", ":")), node_id))
            target.commit()
    return edit


def reset(db: str = DB) -> None:
    """Delete the store so the next run starts from an empty tree."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass
        except PermissionError:
            raise RuntimeError(f"{db} is still open in this process: close the store or restart the kernel, "
                               "then reset again") from None
    reset_counters()


def root_id_by_hand(label: str = RUN_LABEL) -> str:
    """Recompute a root ID with the standard library alone."""
    node = {"a": 0, "k": "root", "p": "", "pl": {"run": label}}
    canonical = json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(b"pollard/v1\n" + canonical).hexdigest()


def pollard_cli(*arguments: str) -> int:
    """Run the pollard command line in this process, for example pollard_cli("verify", DB). Returns the exit code."""
    from pollard.cli import main

    try:
        return int(main([str(argument) for argument in arguments]))
    except SystemExit as stop:  # argparse exits on --help and on a bad argument
        return stop.code if isinstance(stop.code, int) else 1


__all__ = [
    "DB", "RUN_LABEL", "SCRIPTED", "SCRIPTED_MODEL", "SYSTEM", "TASK", "LIVE", "ModelPath",
    "agent", "approve", "assistant_message", "choose_model", "describe", "evidence", "first_messages",
    "has_side_effects", "live_calls", "make_registry", "parse_decision", "pollard_cli", "recorded_model",
    "registry", "reset", "reset_counters", "root_id_by_hand", "scripted_model", "sentinel", "tamper",
    "tool_message", "verify_tree",
    # re-exported so a notebook needs one import line
    "ActionSpec", "ApprovalPolicy", "ApprovalRequest", "Budget", "BudgetExceeded", "IntegrityError",
    "MissingRecording", "PolicyViolation", "Registry", "Runtime", "SQLiteStore", "decide_approval",
    "request_approval", "seal", "verify",
]
