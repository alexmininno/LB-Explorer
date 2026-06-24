# Handoff Log

> [!IMPORTANT]
> REMINDER: Always read this file and `skills.md` at the start of a session. Always append to these files at the end of a session before marking the task complete. Always propose implementation plans for any requests other than updating these handoff files. Always keep the user updated on thoughts and ask for confirmation instead of assuming.

## Date: 2026-06-03

### Completed Work
- Added `argparse` with detailed `--help` descriptions to all standalone scripts (`check_validity_solutions.py`, `check_equivariance.py`, `check_polystability.py`, `check_lb_spec.py`, `generate_sym_groups.py`).
- Standarized script inputs to read from a single database file, replacing hardcoded paths with a configurable `--db_path` defaulting to `databases/full_cicy_database.json`.
- Handled the `sagemath` dependency in `generate_sym_groups.py` gracefully so its help menu remains accessible without SageMath installed.
- Created a new script `scripts/create_LB-Explorer_inputs.py` which takes `databases/full_cicy_database.json` and cleanly formats its data into `cy_geometry_exports/all_geometry_h11_{h11}.json` for `LB-Explorer.py` to ingest.
- Updated `LB-Explorer.py` with `--db_dir` argument to seamlessly consume these generated geometry inputs without needing structural logic changes.
- Fixed an `argparse` formatting crash (`%` literal bug) in `LB-Explorer.py`.
- Fixed a missing module import (`generate_diversity_stats` -> `check_equivariance`) in `check_polystability.py`.
- Wrote `README.md` containing installation steps, explicit `sagemath` requirements, and script usage guides.
- Implemented `.gitignore` to keep logs, caches, and `Handoff/` out of version control if initialized.
- Cleaned up obsolete references to `gpu_only.py` and `_gpu_` filename prefixes in `LB-Explorer.py` documentation and file generation code to fully match the new naming conventions.

### Current Project State
The repo is cleanly organized. The main ML pipeline (`LB-Explorer.py`) and all evaluation scripts (`scripts/*.py`) are fully standalone and well-documented through `argparse`. All paths are relative, parameterizable, and sensibly defaulted to the `databases/` and `Sol_Runs/` folders.

### Pending Blockers
None.

### Explicit Next Steps
- Verify that users can successfully parse the database with `create_LB-Explorer_inputs.py` on their local setup.
- Any future logic changes inside `LB-Explorer.py` should ensure that the expected input format matches what `create_LB-Explorer_inputs.py` produces.
