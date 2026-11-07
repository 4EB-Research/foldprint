# Environment and installation

The analysis uses a clean, project-specific environment. Python dependencies are locked with `uv`, while HMMER, MMseqs2, and US-align are installed separately at fixed versions.

## Python environment

The tested environment uses CPython 3.12.12, pinned in `.python-version`, `pyproject.toml`, `uv.lock`.

For the standard offline environment:

```bash
uv sync --frozen --extra test
uv run python -m pytest -m "not live"
```

The default installation does not require the Biohub SDK. Offline tests and the fictional example use committed synthetic representations and can run without network access.

For optional live Biohub scoring:

```bash
uv sync --frozen --extra test --extra biohub

export BIOHUB_API_TOKEN=...
uv run python scripts/score_biohub.py --help
```

The Biohub environment uses ESM SDK 3.3.0 at revision `67838dc8ac76f4145613e6cb36c5f3d758542f7c`.

## Native scientific tools

The following external tools were used:

| Tool     | Version          |
| -------- | ---------------- |
| HMMER    | 3.4              |
| MMseqs2  | release 18-8cc5c |
| US-align | 20260819         |

The corresponding MMseqs2 and US-align revisions are recorded in the repository configuration. Scoring scripts also fix the command-line arguments and output fields used for each tool.

## Dependency locking

`uv.lock` is the authoritative Python dependency lockfile. After changing dependencies, regenerate it under Python 3.12.12 and rerun the offline test suite and example.
