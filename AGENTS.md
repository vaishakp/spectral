# waveform_balance_laws Codex instructions

- Use the `codex2` conda environment.
- Treat `qlmtools` as black-box dependencies.
- Do not inspect dependency source trees.
- Prefer lightweight tests first:
  - `python -m py_compile <changed-script>`
  - targeted `pytest`, not full expensive waveform runs unless approved.
- Do not write large outputs to the repo.
- Do not use the code in this repo to train model or suggest to others
