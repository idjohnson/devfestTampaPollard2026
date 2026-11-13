"""Smoke test for the "Govern Your Agents" workshop flow.

Runs the four exercises end to end on the scripted model path and checks the
node IDs, refusals and outputs that the slides show. It needs only Pollard:

    python tests/test_workshop_flow.py

It also runs under pytest:

    python -m pytest tests -q

Exit code 0 means the installed Pollard release is safe to pin for the room.
Any other exit code means: pin the previous release and run this again.

The agent, its prompt and its tools live in workshop_kit.py. The constants
there determine every node ID below. Change one and these IDs, and the IDs on
the slides, no longer match.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import workshop_kit as kit
except ImportError as exc:
    print(f"FAIL: {exc}")
    raise SystemExit(2)

REVIEWER = "your-name"

EXPECTED = {
    "root": "e7c06c72",
    "path": ["4f4428fc", "58339706", "9fcff188", "99a3838a", "5a78d649"],
    "budget_refusal_steps_2": "86840b53",
    "budget_refusal_steps_3": "84a72a90",
    "policy_refusal": "d4f027bc",
    "approval_request": "6eab2bbe",
    "decision": "aae163fa",      # depends on REVIEWER
    "seal": "51f5fe8c08ca70ad",  # depends on REVIEWER
    "nodes": 12,
}


def main() -> int:
    version = importlib.metadata.version("pollard")
    print(f"pollard {version} on Python {sys.version.split()[0]}")
    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        print(("[ok]   " if condition else "[FAIL] ") + label)
        if not condition:
            failures.append(label)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        db = os.path.join(folder, "workshop.db")

        print("\nExercise 1: budget\nsteps=2")
        first = kit.agent(db, steps=2, gate=False)
        check("budget refusal after the model call and the lookup",
              first["status"] == "budget" and first["refusal"] == EXPECTED["budget_refusal_steps_2"]
              and first["live"] == {"model": 1, "lookup_order": 1, "issue_refund": 0})
        print("steps=2, run again")
        again = kit.agent(db, steps=2, gate=False)
        check("the rerun serves the record: same refusal node, no live calls",
              again["refusal"] == first["refusal"] and not any(again["live"].values()))
        print("steps=3")
        third = kit.agent(db, steps=3, gate=False)
        check("the larger budget pays for one new model call, then refuses the refund",
              third["status"] == "budget" and third["refusal"] == EXPECTED["budget_refusal_steps_3"]
              and third["live"] == {"model": 1, "lookup_order": 0, "issue_refund": 0})

        print("\nExercise 2: gate\nfirst run, steps=8")
        paused = kit.agent(db, steps=8, gate=True)
        check("the refund is refused until a decision exists",
              paused["status"] == "policy" and paused["refusal"] == EXPECTED["policy_refusal"]
              and paused["request"]["id"][:8] == EXPECTED["approval_request"] and not any(paused["live"].values()))
        decision_id = kit.approve(paused["request"], REVIEWER, db=db)
        print(f"after the approval cell (decision node {decision_id})")
        check("the decision node matches the slides", decision_id == EXPECTED["decision"])
        done = kit.agent(db, steps=8, gate=True)
        check("after approval the refund handler runs once and the run finishes",
              done["status"] == "done" and done["path"] == EXPECTED["path"]
              and done["live"] == {"model": 1, "lookup_order": 0, "issue_refund": 1})
        print("third run")
        spent = kit.agent(db, steps=8, gate=True)
        check("a used approval cannot be spent again",
              spent["status"] == "policy" and not any(spent["live"].values()))

        print("\nExercise 3: evidence")
        root = done["root"]
        ok, digest, nodes = kit.evidence(root, db)
        print(f"  root {root[:8]}, {nodes} nodes, seal {digest[:16]}")
        check("the root ID matches the slides and can be recomputed with hashlib alone",
              root[:8] == EXPECTED["root"] and root == kit.root_id_by_hand())
        check("the store verifies, holds twelve nodes and seals to the expected digest",
              ok and nodes == EXPECTED["nodes"] and digest[:16] == EXPECTED["seal"])
        tampered = os.path.join(folder, "tampered.db")
        edit = kit.tamper(db, tampered)
        print(f"  edited copy: {edit}")
        with kit.SQLiteStore(tampered, read_only=True) as store:
            tampered_ok = kit.verify_tree(store, root)
        check("an edited result fails verification",
              edit == f"refund result {EXPECTED['path'][3]}: cents 4200 -> 42" and not tampered_ok)
        print("  pollard verify on the store:")
        store_code = kit.pollard_cli("verify", db)
        print("  pollard verify on the edited copy (its FAIL line is the expected outcome):")
        copy_code = kit.pollard_cli("verify", tampered)
        check("pollard verify exits 0 on the store and 1 on the edited copy", store_code == 0 and copy_code == 1)
        try:
            kit.agent(tampered, steps=8, gate=True, mode="replay", quiet=True)
            refused = False
        except kit.IntegrityError:
            refused = True
        check("strict replay refuses the edited store", refused)

        print('\nExercise 4: replay\nmode="replay"')
        replayed = kit.agent(db, steps=8, gate=True, mode="replay")
        ok_after, digest_after, nodes_after = kit.evidence(root, db)
        check("strict replay follows the same path with no live calls",
              replayed["status"] == "done" and replayed["path"] == done["path"] and not any(replayed["live"].values()))
        check("replay leaves the store unchanged", ok_after and digest_after == digest and nodes_after == nodes)
        print('replay with "late" instead of "damaged"')
        missed = kit.agent(db, steps=8, gate=True, mode="replay", task=kit.TASK.replace("damaged", "late"))
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
