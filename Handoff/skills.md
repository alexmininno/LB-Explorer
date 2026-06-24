# Skills and Procedures

## Date: 2026-06-03

### New Procedures Established
- **Data Pipeline**: The raw `databases/full_cicy_database.json` should no longer be manipulated or hardcoded into the RL trainer directly. We developed a decoupling pattern:
  1. `create_LB-Explorer_inputs.py` ingests the master database and builds isolated, `h11`-specific geometry configuration files (e.g. `all_geometry_h11_4.json`) within a target directory (default: `cy_geometry_exports`).
  2. `LB-Explorer.py` now accepts `--db_dir` to load geometry directly from this directory without requiring code changes to its ingestion logic.
  
- **CLI Standardization**: 
  - All scripts now rely on `argparse`. 
  - When referencing the master database, the standard argument is `--db_path` defaulting to `databases/full_cicy_database.json`.
  - When referencing input directories containing solutions (`raw_cy_*`), the standard argument is `--input_dir`.
  - When generating plots or outputs, they dynamically switch between normal and transfer learning outputs using `--transfer`.
  - The `LB-Explorer.py` script now generates output files without the `_gpu_` prefix (e.g. `solutions_h11_4_idx_0.jsonl`) to properly reflect its generic structure.

- **SageMath Dependency**: `generate_sym_groups.py` uses `sage.all`. The import is now wrapped in a `try...except ImportError:` block so that the script can still display its `--help` menu gracefully without a fatal Python crash in non-Sage environments.

### Environment Context
- The project environment is named `cy-explorer` (conda).
- SageMath is required for group theory operations but is considered an external system-level dependency.
