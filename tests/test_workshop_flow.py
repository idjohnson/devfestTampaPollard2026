"""Smoke test for the "Govern Your Agents" workshop flow.

Runs the four exercises end to end on the scripted model path and checks the
node IDs, refusals and outputs that the slides show. It needs only Pollard:

    python tests/test_workshop_flow.py

It also runs under pytest:

    python -m pytest tests -q

Exit code 0 means the installed Pollard release is safe to pin for the room.
Any other exit code means: pin the previous release and run this again.

The constants below (run label, prompts, tool specifications, scripted
decisions) determine every node ID. Change one and the IDs on the slides no
longer match.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import sys
import tempfile

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
    print(f"FAIL: this workshop needs pollard 1.6.1 or newer ({exc})")
    raise SystemExit(2)

# ---------------------------------------------------------------- constants
RUN_LABEL = "refund-agent"
MODEL_NAME = "demo-model"
SYSTEM = 'You are a support agent. Reply with one JSON object: {"tool":name,"args":{...}} or {"final":text}.'
TASK = "Order A-1001 arrived damaged. Please refund it."
SHOW = {"lookup_order": ("status",), "issue_refund": ("refund_id", "cents")}

EXPECTED = {
    "root": "e7c06c72",
    "path": ["d6d55ce4", "f08f8e36", "664189cb", "2fb1adff", "03a3e5eb"],
    "budget_refusal_steps_2": "81ea21d5",
    "budget_refusal_steps_3": "1c66f238",
    "policy_refusal": "c259c543",
    "approval_request": "9fae8295",
    "nodes": 12,
}

LIVE = {"model": 0, "lookup_order": 0, "issue_refund": 0}


# ------------------------------------------------------------ model and tools
def scripted_model(payload):
    """Deterministic stand-in for a model: one decision per completed tool call."""
    LIVE["model"] += 1
    done = sum(1 for m in payload["messages"] if m["role"] == "tool")
    decision = [
        {"tool": "lookup_order", "args": {"order_id": "A-1001"}},
        {"tool": "issue_refund", "args": {"order_id": "A-1001", "cents": 4200}},
        {"final": "Refund of $42.00 issued for order A-1001."},
    ][done]
    text = json.dumps(decision, sort_keys=True)
    prompt_chars = sum(len(m["content"]) for m in payload["messages"])
    return {"text": text, "usage": {"input_tokens": prompt_chars // 4, "output_tokens": len(text) // 4}}


def sentinel(_payload):
    raise AssertionError("strict replay must never call a model or a tool")


def lookup_order(args):
    LIVE["lookup_order"] += 1
    return {"order_id": args["order_id"], "status": "delivered", "paid_cents": 4200,
            "usage": {"input_tokens": 0, "output_tokens": 0}}


def issue_refund(args):
    LIVE["issue_refund"] += 1
    return {"refund_id": "R-77", "cents": args["cents"], "usage": {"input_tokens": 0, "output_tokens": 0}}


def make_registry(replay=False):
    def obj(properties, required):
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    return Registry([
        ActionSpec("lookup_order", "1", "Read one order",
                   obj({"order_id": {"type": "string"}}, ["order_id"]),
                   False, sentinel if replay else lookup_order),
        ActionSpec("issue_refund", "1", "Return money to a customer",
                   obj({"order_id": {"type": "string"},
                        "cents": {"type": "integer", "minimum": 1, "maximum": 10000}}, ["order_id", "cents"]),
                   True, sentinel if replay else issue_refund),
    ])


# ------------------------------------------------------------------ the agent
def agent(db, *, steps, gate, mode="hybrid", task=TASK, quiet=False):
    """One execution of the agent cell. It always starts at the root; hybrid mode serves recorded steps."""
    say = (lambda *a: None) if quiet else print
    for key in LIVE:
        LIVE[key] = 0
    out = {"path": [], "status": None, "refusal": None, "request": None, "root": None}
    replay = mode == "replay"
    with SQLiteStore(db, read_only=replay) as store:
        registry = make_registry(replay)
        runtime = Runtime(store, registry=registry, policies=[ApprovalPolicy(store)] if gate else [], mode=mode)
        model = sentinel if replay else scripted_model
        try:
            with runtime.run(RUN_LABEL, budget=Budget(steps=steps)) as run:
                out["root"] = run.root_id
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
                while True:
                    node = run.model_call({"model": MODEL_NAME, "messages": messages}, fn=model)
                    out["path"].append(node.id[:8])
                    decision = json.loads(node.result["text"])
                    messages = messages + [{"role": "assistant", "content": node.result["text"]}]
                    if "final" in decision:
                        out["status"] = "done"
                        say("  FINAL:", decision["final"])
                        break
                    name, args = decision["tool"], decision["args"]
                    if gate and registry.get(name).side_effects:
                        request = request_approval(run, name, args)
                        out["request"] = request.to_dict()
                    try:
                        result = run.tool_call(name, args)
                    except PolicyViolation as refusal:
                        out["status"], out["refusal"] = "policy", refusal.refusal_id[:8]
                        say(f"  REFUSED by policy: {refusal}")
                        say(f"  refusal node {refusal.refusal_id[:8]}, request {request.id[:8]}")
                        break
                    out["path"].append(result.id[:8])
                    body = {k: v for k, v in result.result.items() if k != "usage"}
                    say(f"  tool {name} -> " + ", ".join(f"{k} {body[k]}" for k in SHOW[name]))
                    messages = messages + [{"role": "tool", "content": json.dumps(body, sort_keys=True)}]
        except BudgetExceeded as refusal:
            out["status"], out["refusal"] = "budget", refusal.refusal_id[:8]
            say(f"  REFUSED by budget: {refusal}")
            say(f"  refusal node {refusal.refusal_id[:8]}")
        except MissingRecording as miss:
            out["status"] = "missing"
            say("  MissingRecording:", str(miss)[:60])
    out["live"] = dict(LIVE)
    calls = ", ".join(f"{k} {v}" for k, v in LIVE.items() if v)
    say("  live calls:", calls or "none")
    return out


def approve(db, request_dict, reviewer):
    with SQLiteStore(db) as store:
        return decide_approval(store, ApprovalRequest.from_dict(request_dict), approved=True, reviewer=reviewer).id[:8]


def verify_tree(store, root):
    """verify() checks one node and its ancestors, so check every node, as `pollard verify` does."""
    return all(verify(store, node.id).ok for node in store.walk(root))


def evidence(db, root):
    with SQLiteStore(db, read_only=True) as store:
        return verify_tree(store, root), seal(store, root).digest, sum(1 for _ in store.walk(root))


# ----------------------------------------------------------------- the checks
def main():
    version = importlib.metadata.version("pollard")
    print(f"pollard {version} on Python {sys.version.split()[0]}")
    failures = []

    def check(label, condition):
        print(("[ok]   " if condition else "[FAIL] ") + label)
        if not condition:
            failures.append(label)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        db = os.path.join(folder, "workshop.db")

        print("\nExercise 1: budget\nsteps=2")
        first = agent(db, steps=2, gate=False)
        check("budget refusal after the model call and the lookup",
              first["status"] == "budget" and first["refusal"] == EXPECTED["budget_refusal_steps_2"]
              and first["live"] == {"model": 1, "lookup_order": 1, "issue_refund": 0})
        print("steps=2, run again")
        again = agent(db, steps=2, gate=False)
        check("the rerun serves the record: same refusal node, no live calls",
              again["refusal"] == first["refusal"] and not any(again["live"].values()))
        print("steps=3")
        third = agent(db, steps=3, gate=False)
        check("the larger budget pays for one new model call, then refuses the refund",
              third["status"] == "budget" and third["refusal"] == EXPECTED["budget_refusal_steps_3"]
              and third["live"] == {"model": 1, "lookup_order": 0, "issue_refund": 0})

        print("\nExercise 2: gate\nfirst run, steps=8")
        paused = agent(db, steps=8, gate=True)
        check("the refund is refused until a decision exists",
              paused["status"] == "policy" and paused["refusal"] == EXPECTED["policy_refusal"]
              and paused["request"]["id"][:8] == EXPECTED["approval_request"] and not any(paused["live"].values()))
        decision_id = approve(db, paused["request"], reviewer="your-name")
        print(f"after the approval cell (decision node {decision_id})")
        done = agent(db, steps=8, gate=True)
        check("after approval the refund handler runs once and the run finishes",
              done["status"] == "done" and done["path"] == EXPECTED["path"]
              and done["live"] == {"model": 1, "lookup_order": 0, "issue_refund": 1})
        print("third run")
        spent = agent(db, steps=8, gate=True)
        check("a used approval cannot be spent again",
              spent["status"] == "policy" and not any(spent["live"].values()))

        print("\nExercise 3: evidence")
        root = done["root"]
        ok, digest, nodes = evidence(db, root)
        print(f"  root {root[:8]}, {nodes} nodes, seal {digest[:16]}")
        check("the root ID matches the slides", root[:8] == EXPECTED["root"])
        check("the store verifies and holds twelve nodes", ok and nodes == EXPECTED["nodes"])
        tampered = os.path.join(folder, "tampered.db")
        shutil.copy(db, tampered)
        with contextlib.closing(sqlite3.connect(tampered)) as connection:
            changed = connection.execute(
                "update nodes set result = replace(result, '\"cents\":4200', '\"cents\":42') "
                "where kind = 'tool_call' and payload like '%issue_refund%'").rowcount
            connection.commit()
        with SQLiteStore(tampered, read_only=True) as store:
            tampered_ok = verify_tree(store, root)
        check("an edited result fails verification", changed == 1 and not tampered_ok)
        try:
            agent(tampered, steps=8, gate=True, mode="replay", quiet=True)
            refused = False
        except IntegrityError:
            refused = True
        check("strict replay refuses the edited store", refused)

        print('\nExercise 4: replay\nmode="replay"')
        replayed = agent(db, steps=8, gate=True, mode="replay")
        ok_after, digest_after, nodes_after = evidence(db, root)
        check("strict replay follows the same path with no live calls",
              replayed["status"] == "done" and replayed["path"] == done["path"] and not any(replayed["live"].values()))
        check("replay leaves the store unchanged", ok_after and digest_after == digest and nodes_after == nodes)
        print('replay with "late" instead of "damaged"')
        missed = agent(db, steps=8, gate=True, mode="replay", task=TASK.replace("damaged", "late"))
        check("a changed input misses instead of going live",
              missed["status"] == "missing" and not any(missed["live"].values()))

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed on pollard {version}")
        return 1
    print(f"PASS: every check held on pollard {version}")
    return 0


def test_workshop_flow():
    """pytest entry point."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
