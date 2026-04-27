# ms-core Testing Strategy Implementation Plan

> **Execution note:** Implement this plan task-by-task and verify each step before proceeding to the next.

**Goal:** Make `ms-core` test ownership, local verification scope, and consumer coordination explicit enough for toolkit and DNP to consume it as a shared library.

**Architecture:** Keep durable policy in `docs/TESTING.md`, enforce selectable test layers through central pytest marker classification, and document submodule/consumer coordination in README and AGENTS. Do not change processing behavior in this PR.

**Tech Stack:** Python, pytest, pyproject pytest config, repo-local documentation.

---

### Task 1: Add marker classification tests

**Files:**
- Create: `tests/testing_markers.py`
- Create: `tests/test_testing_markers.py`
- Modify: `tests/conftest.py`
- Modify: `pyproject.toml`

**Steps:**
1. Add path-based marker classification for `algorithm`, `io`, `pipeline`, `contract`, `hygiene`, and `serial`.
2. Add tests that lock the marker mapping for current files.
3. Register markers and enable `--strict-markers`.
4. Run `python -m pytest tests/test_testing_markers.py -v --tb=short`.

### Task 2: Add durable testing strategy documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/TESTING.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `pyproject.toml`

**Steps:**
1. Document test layers, marker policy, local commands, and change-to-test matrix.
2. Document consumer coordination for toolkit and DNP.
3. Add repo-local CI delegation to the shared Python workflow.
4. Align supported CI Python versions with toolkit policy: 3.11 and 3.12.
5. Keep DNP-specific integration out of `ms-core`; describe only shared-library contracts.
6. Read edited docs as UTF-8 text after edits.

### Task 3: Verify and commit `ms-core`

**Files:**
- All changed `ms-core` files.

**Steps:**
1. Run marker tests.
2. Run collect-only to confirm marker registration.
3. Run the focused `ms-core` suite.
4. Commit in `ms-core` first.

### Task 4: Update parent toolkit submodule pointer

**Files:**
- Modify: `ms-core` submodule pointer.

**Steps:**
1. Return to parent repo.
2. Confirm `ms-core` commit is pushed or ready to push before parent merge.
3. Stage only the submodule pointer.
4. Commit parent pointer update.
