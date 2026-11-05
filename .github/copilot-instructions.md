# Copilot instructions

Read `AGENTS.md` and the Master Prompt before changing this repository. The Master Prompt rules R1–R8 are authoritative.
The Master Prompt is `docs/MASTER_PROMPT.md`.

Use `pytest -q tests/unit` for PR-0/PR-1 validation only. Do not download datasets, run full matrices, or start PR-2 or later in an agent session. Preserve leakage isolation, keep canonical EDSP literal, document ambiguities in `docs/SPEC_AMBIGUITIES.md`, and never fabricate results.
