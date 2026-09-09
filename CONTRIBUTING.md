# Contributing to ActionSplice

Bug reports, documentation fixes, backend integrations, and focused code
improvements are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
pytest
```

The CPU test suite does not require upstream world-model repositories or
checkpoints. Backend integration tests may require the separately licensed
dependencies described in `THIRD_PARTY.md`.

## Pull requests

- Open an issue first for behavior changes or new backends.
- Keep changes scoped and include tests for observable behavior.
- Run `ruff check .` and `pytest` before opening the pull request.
- Do not commit checkpoints, generated samples, private paths, credentials, or
  code copied from upstream repositories.
- State which backend, CST variant, Python version, and hardware were tested.

By submitting a contribution, you agree that it is licensed under the
Apache License 2.0 used by this repository.
