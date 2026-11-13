# Changelog

All notable changes to the workshop materials are recorded here.

## Unreleased

### Added

- Repository scaffold: README, pinned requirements, `.env.example`, `.gitattributes`.
- Smoke test for the full exercise flow (`tests/test_workshop_flow.py`), verified on Pollard 1.6.2 and 1.6.1.
- `workshop_kit.py`: the agent's prompt, tools, registry, model paths (scripted, endpoint, colab) and reference loop, shared by the notebook and the tests.
- Unit tests for the kit (`tests/test_workshop_kit.py`).

### Changed

- The system prompt now describes both tools, and tool results go back to the model as user messages, so one payload works on any chat endpoint. Every node ID below the root changed, and the smoke test expects the new IDs.
- The smoke test imports the agent from `workshop_kit.py` and also checks the decision node, the seal digest and the `pollard verify` exit codes.
