# Govern Your Agents: Budgets, Approval Gates, and Exact Replay with Pollard

Materials for a one-hour, hands-on workshop at DevFest, USF Tampa, Saturday, November 14, 2026.

You take a small refund agent, put it under [Pollard](https://github.com/jemsbhai/pollard), and make the runtime say no four ways, on purpose.

| # | Exercise | You cause | What it shows |
|---|---|---|---|
| 1 | Budget | A budget refusal | A refused step never runs; resuming pays only for new steps |
| 2 | Gate | A policy refusal | An approval binds one exact action and is used once |
| 3 | Evidence | A failed verification | The tree is the audit record; an edit is detected |
| 4 | Replay | A replay miss | Strict replay serves the record and never goes live |

## What you need

- A laptop and a browser. The session runs in Google Colab, so a Google account is enough.
- Optional: Python 3.10 or newer, if you prefer to run locally.
- Optional: an API key for an OpenAI-compatible endpoint, if you want a hosted model. A deterministic model path that needs no key is always available.

## Status

| Item | State |
|---|---|
| Smoke test (`tests/test_workshop_flow.py`) | In the repository |
| Notebook (`workshop.ipynb`) | Being added |
| Slides (`slides/`) | Being added |
| Recording of a real model run (`recordings/`) | Being added |

## Run the smoke test locally

The smoke test runs all four exercises on the deterministic model path and checks the node IDs shown on the slides.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tests\test_workshop_flow.py
```

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tests/test_workshop_flow.py
```

The last line should read `PASS: every check held on pollard 1.6.2`. The root ID it prints starts with `e7c06c72`, the same as everyone else's: a root identity depends only on the run label.

## Version policy

The Pollard release is pinned in `requirements.txt`. The pin is the newest release on which the smoke test passes. The floor is 1.6.1, the first release with durable approvals.

## Links

- Pollard: <https://github.com/jemsbhai/pollard>, or `pip install pollard`
- Paper: Syed and Silaghi, "Pollard: Governed Semantic Execution Trees for Framework-Independent AI Agent Control and Replay", IEEE ICTAI 2026, <https://arxiv.org/abs/2610.16145>

## License

MIT. See `LICENSE`.
