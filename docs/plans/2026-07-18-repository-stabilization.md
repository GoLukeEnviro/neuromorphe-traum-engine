# Repository Stabilization Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Consolidate all intended local work and the `feature/service-architecture` history into a verified, clean, GitHub-synchronized `main` branch.

**Architecture:** Treat repository integrity as a sequence of gates: repair Git metadata, reconcile the worktree, remove generated artifacts from version control, verify the application, then publish and integrate. Preserve recoverability by backing up invalid Git metadata and by removing generated files from the index without deleting local copies.

**Tech Stack:** Git, GitHub CLI, PowerShell, Python 3, pytest, flake8, FastAPI, Streamlit, Playwright.

---

### Task 1: Repair and verify Git metadata

**Files:**
- Inspect: `.git/refs/**/__init__.py`
- Inspect: `.git/objects/**/__init__.py`
- Backup: `C:/tmp/neuromorphe-traum-engine-git-init-backup-20260718/`

**Step 1: Reproduce the integrity failure**

Run: `git fsck --full --no-reflogs`

Expected: FAIL with `badRefContent`, `invalid sha1 pointer`, and `bad sha1 file` entries for empty `__init__.py` files.

**Step 2: Prove the cleanup scope**

Run a PowerShell inventory rooted at `.git` and require every matching `__init__.py` to be zero bytes and inside the resolved `.git` path.

Expected: 194 files, zero non-empty files, zero paths outside `.git`.

**Step 3: Move only the proven foreign files to the backup**

Preserve relative paths below the backup root so the operation remains reversible.

**Step 4: Verify Git integrity**

Run: `git fsck --full --no-reflogs`

Expected: exit 0; dangling objects are allowed, invalid refs and bad object-file names are not.

### Task 2: Prevent recurrence in `create_inits.py`

**Files:**
- Modify: `create_inits.py`
- Create: `test/test_create_inits.py`

**Step 1: Write the failing regression test**

Copy the real script into a temporary project containing `.git`, `venv`, `src/node_modules`, and `src` directories. Assert that only the project source directory receives `__init__.py`.

**Step 2: Run the test and verify RED**

Run: `python -m pytest test/test_create_inits.py -v`

Expected: FAIL because the current unrestricted `os.walk(project_root)` writes into excluded trees.

**Step 3: Implement the minimal root-cause fix**

Prune metadata, dependency, environment, cache, and generated-data directories through `dirnames[:]` before descending. Put execution behind `if __name__ == "__main__":` and expose a small `create_init_files()` function.

**Step 4: Run the regression test and verify GREEN**

Run: `python -m pytest test/test_create_inits.py -v`

Expected: all parameter cases PASS.

### Task 3: Reconcile the existing worktree

**Files:**
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Review: `ai_agents/minimal_preprocessor.py`
- Review: `ai_agents/prepare_dataset_sql.py`
- Review: `ai_agents/search_engine_cli.py`
- Add: `.flake8`
- Add: `backend/app.py`
- Preserve locally/ignore: `.claude/plans/jo-verschaffe-dir-einen-curried-yeti.md`

**Step 1: Remove conflict markers from configuration**

Keep the stricter existing dependency versions, add only missing runtime packages, and add project-specific ignore rules for dependency trees and generated runtime data.

**Step 2: Preserve production CLAP behavior**

Attempt to import and initialize real CLAP by default; use the dummy implementation only when the optional model import or initialization actually fails.

**Step 3: Scan for unresolved conflict markers**

Run: `rg -n "^(<<<<<<<|=======|>>>>>>>)" -g '!venv/**' -g '!src/node_modules/**' .`

Expected: no Git conflict markers.

### Task 4: Remove generated artifacts from version control

**Files:**
- Untrack: `src/node_modules/`
- Untrack: `ai_agents/__pycache__/prepare_dataset_sql.cpython-311.pyc`
- Untrack: `processed_database/stems.db`
- Untrack: `processed_database/stems/*.wav`
- Keep: `src/package.json`
- Keep: `src/package-lock.json`

**Step 1: Confirm ignore coverage**

Run `git check-ignore -v` for representative Node, bytecode, database, and generated-audio paths.

Expected: every generated path is covered by a project ignore rule.

**Step 2: Remove artifacts from the index only**

Run targeted `git rm --cached` commands. Do not delete the local dependency installation, database, or audio files.

**Step 3: Verify the tracked set**

Run: `git ls-files | rg '(^|/)(node_modules|__pycache__)(/|$)|\\.(pyc|db)$|^processed_database/stems/'`

Expected: no matches.

### Task 5: Validate the reconciled feature branch

**Files:**
- Test: `tests/`
- Lint: `src/`, `ai_agents/`, `backend/`, `test/test_create_inits.py`

**Step 1: Compile changed Python files**

Run `python -m compileall` against the changed Python modules.

Expected: exit 0.

**Step 2: Run the focused regression test**

Run: `python -m pytest test/test_create_inits.py -v`

Expected: PASS.

**Step 3: Run the full Python suite**

Run: `python -m pytest tests -ra`

Expected: exit 0 with zero failures. If environmental heavyweight dependencies block collection, document the exact blocker and run all independently collectable test groups.

**Step 4: Run lint and import smoke checks**

Run the configured flake8 command and import the FastAPI application.

Expected: exit 0 for every required gate.

### Task 5a: Make application imports independent of optional model downloads

**Files:**
- Modify: `src/audio/service.py`
- Create: `test/test_audio_service_import.py`

**Step 1: Write an import-safety regression test**

Import `audio.service` in a subprocess that fails if `laion_clap` is imported during module initialization.

**Step 2: Run the test and verify RED**

Run: `python -m pytest test/test_audio_service_import.py -v`

Expected: FAIL because `src/audio/service.py` imports `laion_clap` at module scope.

**Step 3: Move the optional import to model initialization**

Import `CLAP_Module` inside `_load_clap_model()` so ordinary application imports remain network- and model-independent. Preserve the existing graceful `None` result when embedding generation cannot load CLAP.

**Step 4: Run the test and verify GREEN**

Run: `python -m pytest test/test_audio_service_import.py -v`

Expected: PASS, followed by a successful full test collection.

**Step 5: Add an application-level import smoke test**

Run `from main import app` in an offline subprocess and assert that the FastAPI title is emitted without model downloads or import-path failures.

### Task 6: Commit and publish the feature branch

**Files:**
- Stage only the reviewed paths shown by `git status --short`.

**Step 1: Review staged scope**

Run: `git diff --check` and `git diff --cached --stat`.

Expected: no whitespace errors; only repository-stabilization changes and generated-file deletions.

**Step 2: Commit**

Run: `git commit -m "chore: stabilize repository and clean generated artifacts"`

Expected: one intentional cleanup commit on `feature/service-architecture`.

**Step 3: Push the feature branch**

Run: `git push -u origin feature/service-architecture`

Expected: remote branch advances to the cleanup commit.

### Task 7: Integrate into `main` and prove final state

**Files:**
- Branch: `main`
- Merge source: `feature/service-architecture`

**Step 1: Switch to and update main**

Run: `git switch main`, then `git pull --ff-only origin main`.

Expected: local `main` matches `origin/main` before integration.

**Step 2: Merge with explicit history**

Run: `git merge --no-ff feature/service-architecture -m "merge: integrate service architecture"`

Expected: the 11 feature commits plus the stabilization commit are reachable from `main`.

**Step 3: Re-run verification on the merge result**

Run the same integrity, focused-test, full-test, compile, and smoke gates from Tasks 1 and 5.

Expected: all required gates pass after the merge.

**Step 4: Push main**

Run: `git push origin main`

Expected: GitHub default branch advances to the verified merge commit.

**Step 5: Clean local branch state**

Run: `git branch -d feature/service-architecture`, then inspect `git status --short --branch`, `git branch -vv`, and `git fsck --full --no-reflogs`.

Expected: clean worktree on synchronized `main`; no unmerged local feature branch; Git integrity exit 0.

---

## Execution record and merge gate

- Git metadata repair: PASS (`git fsck --full --no-reflogs`, exit 0; only pre-existing dangling objects remain).
- Full pytest collection: PASS (343 tests collected offline).
- Repository-maintenance regressions: PASS (7/7 tests).
- Changed-file syntax compilation: PASS.
- Full historical pytest suite: BLOCKED (51 passed, 1 skipped, 219 failed, 72 setup errors).
- Representative baseline mismatch: `tests/conftest.py` calls `create_tables(engine)`, while the current async database API accepts no positional engine argument.
- Main integration rule: do not merge or push `main` until the historical test/API mismatch is repaired or the repository owner explicitly adopts a narrower merge policy.
