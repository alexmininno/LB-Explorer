# LB-Explorer

LB-Explorer is a pure GPU-accelerated PPO agent for discovering holomorphic line bundle sums on Calabi-Yau threefolds.

## Installation and Requirements

The primary environment for running the explorer and its evaluation scripts is defined by `requirements.txt`.
You can create a conda environment as follows:

```bash
conda create -n cy-explorer python=3.10
conda activate cy-explorer
pip install -r requirements.txt
```

### SageMath Dependency
The script `scripts/generate_sym_groups.py` explicitly requires **SageMath** to compute topological and configuration symmetry groups for CICYs. Since SageMath is a large system dependency, it is **not** included in the standard `requirements.txt`.

If you wish to run `generate_sym_groups.py`, you must ensure SageMath is installed on your system and run the script from a SageMath-enabled Python environment (or by invoking `sage -python scripts/generate_sym_groups.py`).

## Database and Inputs

The code uses a unified database structure to maintain Calabi-Yau data.
By default, the master database should be located at `databases/full_cicy_database.json`.

Before running the reinforcement learning agent, you must parse the master database into the expected geometry format. A helper script is provided to generate these inputs automatically:

```bash
python scripts/create_LB-Explorer_inputs.py --db_path databases/full_cicy_database.json --output_dir cy_geometry_exports
```

This will populate `cy_geometry_exports/` with `all_geometry_h11_{h11}.json` files which are required by `LB-Explorer.py`.

## Running the Explorer

You can run the main agent using the `LB-Explorer.py` script. It features a comprehensive Command Line Interface:

```bash
python LB-Explorer.py --h11 6 --cy_index 0 --episodes 10000000 --batch_size 8192 --db_dir cy_geometry_exports
```

For a full list of hyperparameters, reward weights, and architectures, run:
```bash
python LB-Explorer.py --help
```

## Post-Processing Scripts

All evaluation and filtering scripts located in `scripts/` are designed to be run standalone and have an explicit CLI interface with sensible defaults.

### 1. Validity Check
Checks mathematical validity and bounds.
```bash
python scripts/check_validity_solutions.py --help
```

### 2. Equivariance Check
Checks if line bundle solutions are equivariant under freely acting symmetries.
```bash
python scripts/check_equivariance.py --help
```

### 3. Exact Polystability
Verifies exact poly-stability (existence of positive Kahler parameter vectors).
```bash
python scripts/check_polystability.py --help
```

### 4. Line Bundle Spectrum
Filters solutions according to exact spectrum requirements.
```bash
python scripts/check_lb_spec.py --help
```

### 5. Symmetry Generation (Requires SageMath)
Computes symmetry groups for canonicalization.
```bash
python scripts/generate_sym_groups.py --help
```

## Handoff Information
You can find more detailed task progressions and updates in `Handoff/handoff.md` and `Handoff/skills.md`.
