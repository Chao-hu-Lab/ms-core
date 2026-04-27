# ms-core Testing Strategy

This document is the source of truth for `ms-core` test ownership, local test
selection, and downstream consumer coordination.

`ms-core` is a shared library. It should protect processing behavior and public
contracts without depending on any consumer GUI or workflow project.

## Goals

- Keep verification proportional to the changed surface.
- Make shared-library boundaries explicit for toolkit, DNP, and future consumers.
- Keep algorithm tests close to `ms-core` instead of duplicating them in consumer repos.
- Preserve Windows-friendly root hygiene by using repo-local temp fixtures.
- Prevent hidden pytest marker drift through `--strict-markers`.

## Ownership Boundaries

| Surface | Owner | Test location | Notes |
| --- | --- | --- | --- |
| Core data model | `ms-core` | `tests/test_dataset.py` | Public `MSDataset` behavior and metadata handling |
| Processing algorithms | `ms-core` | `tests/test_*feature*`, `tests/test_*organizer*` | Pure processing behavior and scientific regressions |
| Pipeline orchestration | `ms-core` | `tests/test_pipeline.py` | Step registry, dispatch, snapshots, in-memory calibration path |
| Storage / IO contract | `ms-core` | `tests/test_intermediate_store.py`, `tests/test_cache_path_policy.py` | Parquet cache, Excel fallback, intermediate metadata |
| Root hygiene | `ms-core` | `tests/test_root_hygiene.py` | Temp/cache policy and cleanup expectations |
| Consumer bridge | Consumer repo | toolkit/DNP tests | Import paths, GUI behavior, packaging, and app-specific wiring |

Consumer repositories should not add new `ms-core` algorithm tests unless they
are temporarily locking a bridge migration. Shared behavior belongs here first.

## Marker Policy

Marker assignment is centralized in `tests/testing_markers.py` and applied by
`tests/conftest.py` during collection. Do not scatter one-off marker decorators
unless a test has behavior that cannot be expressed by file ownership.

Registered markers:

- `algorithm`: core preprocessing, calibration, statistics, or feature filtering behavior.
- `contract`: public `ms-core` data model, pipeline, or storage contract consumed downstream.
- `hygiene`: repository temp/cache/root hygiene checks.
- `io`: file, cache, parquet, Excel, or intermediate storage behavior.
- `pipeline`: pipeline orchestration and step dispatch behavior.
- `serial`: tests that touch shared repo state or should not run concurrently.

`pyproject.toml` uses `--strict-markers`, so every custom marker must be
registered before use.

## Local Commands

Use PowerShell from the `ms-core` repo root.

### Marker mapping

```powershell
python -m pytest tests/test_testing_markers.py -v --tb=short
```

### Fast public contract check

```powershell
python -m pytest -m contract -v --tb=short
```

### Algorithm-focused

```powershell
python -m pytest -m algorithm -v --tb=short
```

### Storage / IO-focused

```powershell
python -m pytest -m io -v --tb=short
```

### Pipeline-focused

```powershell
python -m pytest -m pipeline -v --tb=short
```

### Root hygiene

```powershell
python -m pytest -m hygiene -v --tb=short
```

### Full `ms-core` suite

```powershell
python -m pytest tests/ -v --tb=short -x
```

### Collection check after marker changes

```powershell
python -m pytest --collect-only tests -q
```

## Change-To-Test Matrix

| Change | Minimum verification | Expand when |
| --- | --- | --- |
| Documentation only | Read affected docs as UTF-8 | README or agent instructions changed |
| `pyproject.toml` pytest config or marker mapping | `tests/test_testing_markers.py`, collect-only | Marker behavior changed |
| `MSDataset` or public model behavior | `-m contract` | Consumer adapters may be affected |
| Feature filtering, data organizer, calibration, statistics | `-m algorithm` plus focused file | Scientific output changes |
| Pipeline registry, step dispatch, snapshots | `-m pipeline` | Public pipeline contract changes |
| Parquet, Excel, cache, intermediate store | `-m io` | Consumer export/import behavior changes |
| Temp fixtures or cleanup behavior | `-m hygiene` | pytest/root behavior changed |
| Public API or schema behavior | `-m contract`, then consumer adapter tests in affected repo | toolkit/DNP imports or output schemas changed |

Start with the narrowest relevant command, then expand only when the changed
surface is shared, public, or risky.

## Root Hygiene Rules

- Do not create temp directories in the repository root.
- Use `project_temp_dir` or `project_temp_root` from `tests/conftest.py`.
- The repo-local test temp root is `.tmp/tests/`.
- Do not re-enable pytest's cache provider unless this policy is updated and
  verified.
- Treat `hygiene` tests as `serial` until their shared-state behavior is fully
  isolated.

## Consumer Coordination

`ms-core` should be consumed by pinned commit or tag. Toolkit, DNP, and future
projects should bump intentionally instead of assuming they are always in sync.

When a change touches public data shapes, function signatures, pipeline steps,
or intermediate storage metadata:

1. Add or update `ms-core` contract tests first.
2. Run the focused `ms-core` tests from this document.
3. Note affected consumers in the PR description.
4. Update each consumer's submodule pointer or dependency pin in a separate PR.
5. Run that consumer's adapter/bridge tests after the bump.

`ms-core` should not import toolkit, DNP, or project-specific GUI code. Consumer
projects own their own GUI, launch flow, packaging, and workflow copy.

## Adding Tests

Before adding a test:

1. Pick the owning layer from the table above.
2. Put shared algorithm/internal behavior in `ms-core/tests/`.
3. Use existing temp fixtures before creating new setup.
4. If the file should be selected by a marker, update `tests/testing_markers.py`
   and `tests/test_testing_markers.py`.
5. Add a new marker only if it changes selection or scheduling.
6. Run the narrowest relevant command from this document.

## CI Contract

Repo-local CI is defined in `.github/workflows/ci.yml` and delegates execution to
`Chao-hu-Lab/shared-workflows/.github/workflows/python-ci.yml@main`.

Current repo-owned expectations:

- CI checks PRs targeting `master` or `main`.
- Push CI covers `master`, `main`, `feature/*`, `fix/*`, and `chore/*`.
- CI runs Python 3.11 and 3.12.
- CI sets `pythonpath: "src"` so tests import the local checkout.

If the shared workflow changes its pytest command, update this document or add a
repo-local wrapper script so local and CI behavior do not drift silently.

Task plans may list one-off verification commands, but this file remains the
durable testing policy.
