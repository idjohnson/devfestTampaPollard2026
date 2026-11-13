"""Unit tests for workshop_kit: reply parsing, model paths, odd model replies, approvals, tampering and reset.

    python -m pytest tests -q

Nothing here reaches the network. The endpoint test talks to a stub server on localhost and is skipped
when the openai package is not installed. The Colab test uses a stand-in module, so it checks the kit's
plumbing and not the Colab service itself.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workshop_kit as kit  # noqa: E402

LOOKUP = '{"tool": "lookup_order", "args": {"order_id": "A-1001"}}'
REFUND = '{"tool": "issue_refund", "args": {"order_id": "A-1001", "cents": 4200}}'
FINAL = '{"final": "Refund issued."}'
FAKE_KEY = "not-a-real-key-0123456789"
SETTINGS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "WORKSHOP_MODEL", "WORKSHOP_MODEL_PATH")
NO_LIVE_TOOLS = {"model": 1, "lookup_order": 0, "issue_refund": 0}
FULL_RUN = {"model": 3, "lookup_order": 1, "issue_refund": 1}


def canned(*replies: str, name: str = "canned-model") -> kit.ModelPath:
    """A model path that returns fixed replies in order. A call past the last reply fails the test."""
    remaining = list(replies)

    def fn(payload):
        kit.LIVE["model"] += 1
        return {"text": remaining.pop(0), "usage": {"input_tokens": 1, "output_tokens": 1}}

    return kit.ModelPath("canned", name, fn, "test stand-in")


def reply_for(sent: str) -> str:
    """What a model that follows the protocol answers, given everything it was sent."""
    return [LOOKUP, REFUND, FINAL][min(sent.count(kit.TOOL_RESULT_PREFIX), 2)]


def first_model_call(db: str, root: str):
    with kit.SQLiteStore(db, read_only=True) as store:
        return next(node for node in store.walk(root) if node.kind == "model_call")


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "workshop.db")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No workshop settings in the environment and an empty working directory, so no local .env or key leaks in.

    Each setting is registered with monkeypatch first, so a value the kit writes during a test is undone afterwards.
    """
    for key in SETTINGS:
        monkeypatch.setenv(key, "registered")
        monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    return os.environ


# ------------------------------------------------------------- messages and parsing
@pytest.mark.parametrize("text, expected", [
    (LOOKUP, {"tool": "lookup_order", "args": {"order_id": "A-1001"}}),
    ("```json\n" + REFUND + "\n```", {"tool": "issue_refund", "args": {"order_id": "A-1001", "cents": 4200}}),
    ("Here is my answer: " + FINAL + " Thank you.", {"final": "Refund issued."}),
    ('{"final": "Done.", "confidence": "high"}', {"final": "Done."}),
])
def test_parse_decision_reads_the_json_object(text, expected):
    assert kit.parse_decision(text) == expected


@pytest.mark.parametrize("text, reason", [
    ("", "holds no JSON object"),
    ("I will look up the order now.", "holds no JSON object"),
    ("{not json}", "not valid JSON"),
    ('{"tool": "lookup_order"}', "neither a tool call nor a final answer"),
    ('{"final": 3}', "neither a tool call nor a final answer"),
])
def test_parse_decision_rejects_other_replies(text, reason):
    with pytest.raises(ValueError, match=reason):
        kit.parse_decision(text)


def test_messages_use_plain_chat_roles():
    result = kit.lookup_order({"order_id": "A-1001"})
    messages = kit.first_messages() + [kit.assistant_message(LOOKUP), kit.tool_message("lookup_order", result)]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == kit.SYSTEM and messages[1]["content"] == kit.TASK
    assert messages[3]["content"] == ('Tool result for lookup_order: '
                                      '{"order_id": "A-1001", "paid_cents": 4200, "status": "delivered"}')


def test_scripted_model_makes_one_decision_per_tool_result():
    messages, decisions = kit.first_messages(), []
    for _ in range(4):
        reply = kit.scripted_model({"model": kit.SCRIPTED_MODEL, "messages": messages})
        assert reply["usage"]["input_tokens"] > 0 and reply["usage_estimated"] is True
        decisions.append(kit.parse_decision(reply["text"]))
        messages = messages + [kit.assistant_message(reply["text"]), kit.tool_message("any_tool", {"ok": True})]
    assert [d.get("tool", "final") for d in decisions] == ["lookup_order", "issue_refund", "final", "final"]


def test_describe_and_live_calls_print_what_the_slides_show():
    kit.reset_counters()
    assert kit.live_calls() == "none"
    assert kit.describe("issue_refund", kit.issue_refund({"cents": 4200})) == "refund_id R-77, cents 4200"
    assert kit.describe("other_tool", {"ok": True, "usage": {}}) == '{"ok": true}'
    assert kit.live_calls() == "issue_refund 1"
    kit.reset_counters()


def test_has_side_effects():
    assert kit.has_side_effects("issue_refund") is True
    assert kit.has_side_effects("lookup_order") is False
    assert kit.has_side_effects("delete_everything") is False


def test_root_id_by_hand_matches_the_runtime(db):
    assert kit.agent(db, steps=1, gate=False, quiet=True)["root"] == kit.root_id_by_hand()
    assert kit.root_id_by_hand("another-run") != kit.root_id_by_hand()


# ------------------------------------------------------------------- model paths
def test_choose_model_defaults_to_scripted(clean_env):
    assert kit.choose_model(quiet=True) is kit.SCRIPTED
    assert kit.choose_model("scripted", quiet=True) is kit.SCRIPTED


def test_choose_model_reads_the_path_from_an_env_file(clean_env, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("# local settings\nWORKSHOP_MODEL_PATH=colab\nWORKSHOP_MODEL='quoted-name'\n",
                                   encoding="utf-8")
    stand_in = canned(FINAL)
    monkeypatch.setitem(kit._PATHS, "colab", lambda model_name: stand_in)
    assert kit.choose_model(quiet=True) is stand_in
    assert clean_env["WORKSHOP_MODEL"] == "quoted-name"


def test_choose_model_falls_back_and_keeps_the_key_off_the_screen(clean_env, monkeypatch, capsys):
    clean_env["OPENAI_API_KEY"] = FAKE_KEY

    def broken(model_name):
        raise RuntimeError(f"401 for key {FAKE_KEY}")

    monkeypatch.setitem(kit._PATHS, "endpoint", broken)
    assert kit.choose_model("endpoint") is kit.SCRIPTED
    printed = capsys.readouterr().out
    assert "Falling back to the scripted path." in printed and "Active model path: scripted" in printed
    assert FAKE_KEY not in printed and "***" in printed


def test_choose_model_rejects_an_unknown_path():
    with pytest.raises(ValueError):
        kit.choose_model("telepathy", quiet=True)


class _StubEndpoint(BaseHTTPRequestHandler):
    """A chat-completions endpoint that follows the workshop protocol and wraps its JSON in a code fence."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        sent = "\n".join(message["content"] for message in body["messages"])
        content = "OK" if sent == "Reply with OK." else "```json\n" + reply_for(sent) + "\n```"
        payload = json.dumps({
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_endpoint(clean_env, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubEndpoint)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clean_env.update(OPENAI_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1", OPENAI_API_KEY=FAKE_KEY,
                     WORKSHOP_MODEL="stub-model")
    for name in ("NO_PROXY", "no_proxy"):  # a configured proxy must not see the request to localhost
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    yield server
    server.shutdown()
    server.server_close()


def test_endpoint_path_records_provider_usage_and_replays_without_the_endpoint(stub_endpoint, db):
    pytest.importorskip("openai", minversion="1.0")
    model = kit.choose_model("endpoint", quiet=True)
    assert (model.path, model.name) == ("endpoint", "stub-model")
    out = kit.agent(db, steps=8, gate=False, model=model, quiet=True)
    assert out["status"] == "done" and out["live"] == FULL_RUN
    assert first_model_call(db, out["root"]).result["usage"] == {"input_tokens": 11, "output_tokens": 7}
    last_request = stub_endpoint.requests[-1]
    assert last_request["model"] == "stub-model"
    assert [m["role"] for m in last_request["messages"]] == ["system", "user", "assistant", "user", "assistant", "user"]
    served = len(stub_endpoint.requests)
    replayed = kit.agent(db, steps=8, gate=False, mode="replay", model=kit.recorded_model(db), quiet=True)
    assert replayed["path"] == out["path"] and len(stub_endpoint.requests) == served


@pytest.fixture
def fake_colab(monkeypatch):
    """A stand-in for google.colab.ai with the two calls the kit uses."""
    ai = types.ModuleType("google.colab.ai")
    ai.prompts = []

    def generate_text(prompt, model_name=None):
        ai.prompts.append((model_name, prompt))
        return "OK" if prompt == "Reply with OK." else reply_for(prompt)

    ai.generate_text = generate_text
    ai.list_models = lambda: ["google/model-flash-lite", "google/model-flash", "google/model-pro"]
    colab = types.ModuleType("google.colab")
    colab.ai = ai
    google = types.ModuleType("google")
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.ai", ai)):
        monkeypatch.setitem(sys.modules, name, module)
    return ai


def test_colab_path_sends_one_flat_prompt_and_marks_usage_as_estimated(fake_colab, clean_env, db):
    model = kit.choose_model("colab", quiet=True)
    assert (model.path, model.name) == ("colab", "google/model-flash")
    out = kit.agent(db, steps=8, gate=False, model=model, quiet=True)
    assert out["status"] == "done" and out["live"] == FULL_RUN
    name, prompt = fake_colab.prompts[1]
    assert name == "google/model-flash"
    assert prompt.startswith("SYSTEM: You are a support agent") and prompt.endswith("\n\nASSISTANT:")
    recorded = first_model_call(db, out["root"]).result
    assert recorded["usage_estimated"] is True and recorded["usage"]["input_tokens"] > 0
    assert kit.choose_model("colab", model_name="google/model-pro", quiet=True).name == "google/model-pro"


def test_recorded_model_names_the_model_in_a_recording(db, tmp_path):
    kit.agent(db, steps=8, gate=False, model=canned(LOOKUP, REFUND, FINAL, name="some-hosted-model"), quiet=True)
    recorded = kit.recorded_model(db)
    assert (recorded.path, recorded.name) == ("recording", "some-hosted-model")
    assert kit.agent(db, steps=8, gate=False, mode="replay", model=recorded, quiet=True)["status"] == "done"
    empty = str(tmp_path / "empty.db")
    with kit.SQLiteStore(empty):
        pass
    with pytest.raises(ValueError):
        kit.recorded_model(empty)


# ------------------------------------------------------------- the reference loop
def test_agent_accepts_replies_wrapped_in_prose_and_serves_them_on_a_rerun(db):
    model = canned("Sure.\n```json\n" + LOOKUP + "\n```", "I will refund it: " + REFUND, FINAL + "\n")
    out = kit.agent(db, steps=8, gate=False, model=model, quiet=True)
    assert out["status"] == "done" and out["live"] == FULL_RUN
    again = kit.agent(db, steps=8, gate=False, model=model, quiet=True)  # no replies left: a live call would fail
    assert again["path"] == out["path"] and not any(again["live"].values())


def test_agent_gives_an_experiment_its_own_label_and_budget(db):
    main = kit.agent(db, steps=2, gate=False, quiet=True)
    side = kit.agent(db, steps=2, gate=False, label="experiment", task="Where is order A-1001?", quiet=True)
    assert main["status"] == side["status"] == "budget"
    assert side["root"] == kit.root_id_by_hand("experiment") != main["root"]
    assert side["live"] == {"model": 1, "lookup_order": 1, "issue_refund": 0}  # a fresh budget, not the spent one


def test_agent_stops_on_a_reply_it_cannot_read(db, capsys):
    model = canned("I will look up the order now.")
    out = kit.agent(db, steps=8, gate=False, model=model)
    assert out["status"] == "protocol" and out["live"] == NO_LIVE_TOOLS
    assert "STOPPED: the model reply holds no JSON object" in capsys.readouterr().out
    again = kit.agent(db, steps=8, gate=False, model=model, quiet=True)
    assert again["status"] == "protocol" and not any(again["live"].values())


def test_agent_stops_on_a_reply_without_text(db):
    silent = kit.ModelPath("canned", "silent-model", lambda payload: {"usage": {"input_tokens": 1, "output_tokens": 0}},
                           "test stand-in")
    assert kit.agent(db, steps=8, gate=False, model=silent, quiet=True)["status"] == "protocol"


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("reply, status, message", [
    ('{"tool": "delete_everything", "args": {}}', "policy", "unknown registered action: delete_everything"),
    ('{"tool": "issue_refund", "args": {"order_id": "A-1001", "cents": 20000}}', "policy",
     "schema validation failed: $.cents: value must be at most 10000"),
    ('{"tool": "issue_refund", "args": {"order_id": "A-1001", "cents": 42.5}}', "rejected",
     "floats are not allowed in identity payloads"),
])
def test_agent_reports_tool_calls_the_runtime_will_not_dispatch(db, capsys, gate, reply, status, message):
    out = kit.agent(db, steps=8, gate=gate, model=canned(reply))
    assert out["status"] == status and out["request"] is None and out["live"] == NO_LIVE_TOOLS
    assert (out["refusal"] is not None) == (status == "policy")
    assert message in capsys.readouterr().out


def test_a_denied_request_keeps_the_refund_from_running(db):
    paused = kit.agent(db, steps=8, gate=True, quiet=True)
    assert paused["status"] == "policy" and paused["request"]["name"] == "issue_refund"
    denial = kit.approve(paused["request"], "reviewer-a", approved=False, db=db)
    assert kit.approve(paused["request"], "reviewer-a", approved=False, db=db) == denial
    with pytest.raises(kit.IntegrityError):
        kit.approve(paused["request"], "reviewer-b", approved=True, db=db)
    again = kit.agent(db, steps=8, gate=True, quiet=True)
    assert again["status"] == "policy" and not any(again["live"].values())


# ------------------------------------------------------- evidence and housekeeping
def test_tamper_edits_a_copy_and_leaves_the_store_alone(db, tmp_path):
    kit.agent(db, steps=8, gate=False, quiet=True)
    root = kit.root_id_by_hand()
    before = kit.evidence(root, db)
    copy = str(tmp_path / "tampered.db")
    assert kit.tamper(db, copy).endswith("cents 4200 -> 42")
    assert before[0] is True and kit.evidence(root, db) == before
    with kit.SQLiteStore(copy, read_only=True) as store:
        assert kit.verify_tree(store, root) is False
        assert kit.verify(store, root).ok is True  # the root alone says nothing about its descendants


def test_tamper_without_a_refund_edits_the_first_model_reply(db, tmp_path):
    kit.agent(db, steps=2, gate=False, quiet=True)
    copy = str(tmp_path / "tampered.db")
    assert kit.tamper(db, copy).startswith("model reply ")
    with kit.SQLiteStore(copy, read_only=True) as store:
        assert kit.verify_tree(store, kit.root_id_by_hand()) is False


def test_tamper_needs_an_existing_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        kit.tamper(str(tmp_path / "missing.db"), str(tmp_path / "copy.db"))


def test_reset_deletes_the_store_and_clears_the_counters(db):
    kit.agent(db, steps=2, gate=False, quiet=True)
    assert os.path.exists(db)
    kit.reset(db)
    assert not os.path.exists(db) and kit.live_calls() == "none"
    kit.reset(db)  # nothing left to delete is fine
    assert kit.agent(db, steps=2, gate=False, quiet=True)["live"]["model"] == 1


def test_pollard_cli_returns_exit_codes(db, capsys):
    kit.agent(db, steps=2, gate=False, quiet=True)
    assert kit.pollard_cli("verify", db) == 0
    assert "OK: 1 roots" in capsys.readouterr().out
    assert kit.pollard_cli("no-such-command") == 2
