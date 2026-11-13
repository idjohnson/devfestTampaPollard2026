# Govern Your Agents: Budgets, Approval Gates, and Exact Replay with Pollard

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jemsbhai/devfestTampaPollard2026/blob/main/workshop.ipynb)

Materials for a one-hour, hands-on workshop at DevFest, USF Tampa, Saturday, November 14, 2026.

You take a small refund agent, put it under [Pollard](https://github.com/jemsbhai/pollard), and make the runtime say no four ways, on purpose.

| # | Exercise | You cause | What it shows |
|---|---|---|---|
| 1 | Budget | A budget refusal | A refused step never runs; a rerun pays only for new steps |
| 2 | Gate | A policy refusal | An approval binds one exact action and is used once |
| 3 | Evidence | A failed verification | The tree is the audit record; an edit is detected |
| 4 | Replay | A replay miss | Strict replay serves the record and never goes live |

## What you need

- A laptop and a browser. The session runs in Google Colab, so a Google account is enough.
- Optional: Python 3.10 or newer, if you prefer to run locally.
- Optional: an API key for an OpenAI-compatible endpoint, if you want a hosted model. A deterministic model path that needs no key is always available.

## Start here

1. Open the notebook in Colab with the badge above.
2. Run the three Setup cells. The third should print `PASS: every check held on pollard 1.6.2`.
3. Work through the four exercises. An exercise cell stops at the first line that still holds a `todo(...)` call: replace the call with the code inside its quotes and run the cell again. A folded Solution cell sits under each of the first two exercises, in case you fall behind.

After the exercises, the notebook runs the same flow on a real model, and then the walkthroughs, offline recipes and offline experiment scripts from the Pollard repository. The ones that need a database, a credential or special hardware are listed with their commands.

## What is in this repository

| File | What it is |
|---|---|
| `workshop.ipynb` | The notebook: setup, the four exercises, a real-model run and the gallery |
| `workshop_kit.py` | What the notebook imports: the agent's prompt, its two tools, the model paths, the finished agent loop and a self-check |
| `requirements.txt` | The pinned Pollard release |
| `.env.example` | Settings for the optional hosted-model path, for local runs |

## Run it locally

Windows (PowerShell):

```powershell
git clone https://github.com/jemsbhai/devfestTampaPollard2026
cd devfestTampaPollard2026
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe workshop_kit.py
```

macOS or Linux:

```bash
git clone https://github.com/jemsbhai/devfestTampaPollard2026
cd devfestTampaPollard2026
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python workshop_kit.py
```

The last command runs the four exercises on the deterministic model path, and its last line should read `PASS: every check held on pollard 1.6.2`. The root ID it prints starts with `e7c06c72`, the same as everyone else's: a root identity depends only on the run label.

Then open `workshop.ipynb` from this folder in Jupyter or VS Code, with that environment as the kernel. Jupyter is not in `requirements.txt`; install `jupyterlab` into the environment if you do not have it.

## Links

- Pollard: <https://github.com/jemsbhai/pollard>, or `pip install pollard`
- Paper: Syed and Silaghi, "Pollard: Governed Semantic Execution Trees for Framework-Independent AI Agent Control and Replay", IEEE ICTAI 2026, <https://arxiv.org/abs/2610.16145>

## License

MIT. See `LICENSE`.
