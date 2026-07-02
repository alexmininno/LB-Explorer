# LB-Explorer

This repository contains the code and tools necessary to reproduce the findings of the paper [arXiv:2607.00078](https://arxiv.org/abs/2607.00078).

## Installation and Requirements

The primary environment for running the explorer and its evaluation scripts is defined by `requirements.txt`.
You can create a conda environment as follows:

```bash
conda create -n cy-explorer python=3.9
conda activate cy-explorer
pip install -r requirements.txt
```

### SageMath Dependency
The script `scripts/generate_sym_groups.py` explicitly requires **SageMath** to compute topological and configuration symmetry groups for CICYs. Since SageMath is a large system dependency, it is **not** included in the standard `requirements.txt`.

If you wish to run `generate_sym_groups.py`, you must ensure SageMath is installed on your system and run the script from a SageMath-enabled Python environment (or by invoking `sage -python scripts/generate_sym_groups.py`). We, however, provide `databases/full_cicy_database.json` with the groups for all CICYs already generated.

## Database and Inputs

The code uses a unified database structure to maintain Calabi-Yau data.
By default, the master database should be located at `databases/full_cicy_database.json`.

### Database Structure

The `databases/full_cicy_database.json` file contains a JSON array of Calabi-Yau threefold geometries and their properties. Each entry is a dictionary containing topological data, intersection polynomials, symmetry group information, and configuration matrices. This database is built based on the one provided by [arxiv:1708.07907](https://arxiv.org/abs/1708.07907).

Here is an example of the entry associated with `Num` 7890:

```json
{
    "Num": 7890,
    "H11": 1,
    "H21": 101,
    "C2": [
        50
    ],
    "Conf": [
        [
            5
        ]
    ],
    "Favour": true,
    "KahlerPos": true,
    "IsProduct": false,
    "Ring": "5*J(1)^3",
    "Group Structure Conf": "1",
    "Group Order Conf": 1,
    "Group Generators Conf": [],
    "Group Structure": "1",
    "Group Order": 1,
    "Group Generators": [],
    "Freely Acting Symmetry": true,
    "Gamma Order": [
        5,
        25
    ],
    "Mapped Freely Acting Symmetries": [
        [
            "C5",
            5,
            []
        ],
        [
            "C25",
            25,
            []
        ]
    ]
}
```

Before running LB-Explorer, you must parse the database into the expected geometry format. A helper script is provided to generate these inputs automatically:

```bash
python scripts/create_LB-Explorer_inputs.py \
    --db_path databases/full_cicy_database.json \
    --output_dir cy_geometry_exports
```

This will populate `cy_geometry_exports/` with `all_geometry_h11_{h11}.json` files which are required by `LB-Explorer.py`.

## Running the Explorer

`LB-Explorer.py` trains a transformer-based PPO agent to generate integer K-matrices whose columns encode the Chern classes of line bundle factors. It evaluates candidates against geometric constraints (anomaly cancellation, Bogomolov stability, chiral indices) and saves valid solutions to JSONL files.

### Arguments for `LB-Explorer.py`:

- `-h, --help`: show this help message and exit
- `--h11`: H11 of the manifold
- `--cy_index`: Index of the CY manifold in database
- `--gamma`: Specific target Gamma value to use.
- `--db_dir`: Directory containing the parsed geometry json files (default: `cy_geometry_exports`)
- `--rank`: Rank of the vector bundle
- `--m_bound`: Max integer charge bound for matrices (default: 8)
- `--stability_range`: Integer range for generating stability test vectors (e.g. default 2 means [-2, 2])
- `--enforce_bounds`: Enforce charge bounds strictly during validation
- `--anom_weight`: Weight for Anomaly Cancellation penalty (default: 1.0)
- `--stab_weight`: Weight for Bogomolov Stability bounds penalty (default: 1.0)
- `--sum_weight`: Weight for Chiral Index Sum penalty (default: 1.0)
- `--rng_weight`: Weight for Chiral Index Range bounds penalty (default: 1.0)
- `--pair_weight`: Weight for Pairwise Index penalty (default: 1.0)
- `--bnd_weight`: Weight for Charge Bounds exceeding penalty (default: 1.0)
- `--anom_coef`, `--stab_coef`, `--sum_coef`, `--rng_coef`, `--pair_coef`, `--bnd_coef`: Coefficient flags for respective penalties (default: 1.0)
- `--embedding_dim`: Embedding dimension for the Transformer (default: 128)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of Transformer layers (default: 4)
- `--episodes`: Total number of RL training episodes (default: 10000000)
- `--batch_size`: Parallel generation batch size (scales with VRAM) (default: 8192)
- `--use_minibatches`: Enable mini-batching during PPO update for stability
- `--minibatch_size`: Size of mini-batches if enabled (default: 1024)
- `--ppo_epochs`: Number of PPO optimization epochs per batch (default: 4)
- `--lr`: Adam Optimizer Learning Rate (default: 0.0003)
- `--entropy_start`: Initial entropy coefficient (Exploration) (default: 0.05)
- `--entropy_end`: Final entropy coefficient (Exploitation) (default: 0.05)
- `--discount`: Gamma discount factor for RL rewards (default: 0.99)
- `--gae_lambda`: Lambda parameter for GAE (default: 0.95)
- `--clip_eps`: PPO Policy clipping parameter (default: 0.2)
- `--vf_coef`: Value Function loss coefficient (default: 0.5)
- `--disable_novelty_penalty`: Turn off penalty for repeating previously generated matrices
- `--novelty_penalty_factor`: Multiplier for score if matrix is a duplicate (e.g. 0.25 cuts score by 75 percent) (default: 0.5)
- `--novelty_buffer_size`: Size of the pure GPU FIFO rolling history buffer (default: 32768)
- `--device`: Compute device to run on (e.g., cuda:0, cpu) (default: cuda:0 if available else cpu)
- `--run_id`: Suffix ID to append to generated output files
- `--output_dir`: Directory to save outputs (default: sol_runs_{run_id} or sol_runs)
- `--resume`: Resume training from existing checkpoint if found
- `--no_plot`: Disable generating matplotlib charts
- `--track_diversity`: Compute batch diversity metric on GPU (off by default)
- `--plot_only`: Only generate charts from checkpoint and exit immediately
- `--no_bonus`: Disable the +5.0 bonus reward for perfect solutions
- `--seed`: Global random seed for reproducibility (default: 42)

**Example Usage**:
```bash
python LB-Explorer.py \
    --h11 5 \
    --cy_index 7447 \
    --gamma 2 \
    --m_bound 8 \
    --stability_range 2 \
    --use_minibatches \
    --no_bonus \
    --run_id h11_5_g2__cy7447__s42 \
    --output_dir sol_runs_h11_5_g2__cy7447__s42
```


## CP-SAT Hybrid Closure

`CPSAT-closure.py` is a script that completes partial integer K-matrices by employing a Constraint Programming SAT solver (CP-SAT). Often, you may want to search for partial matrices by running `LB-Explorer.py` with the chiral index sum penalty disabled (`--sum_coef 0`). Then, you can feed these candidates to `CPSAT-closure.py`, which searches for additional columns that "close" the matrix—satisfying all geometric constraints including anomaly cancellation, exact chiral indices, and Bogomolov stability.

First, generate solutions with `LB-Explorer.py` imposing `--sum_coef 0`. For example:

```bash
python LB-Explorer.py \
    --h11 5 \
    --cy_index 7447 \
    --gamma 2 \
    --m_bound 8 \
    --stability_range 2 \
    --use_minibatches \
    --no_bonus \
    --sum_coef 0 \
    --run_id h11_5_g2__cy7447__s42 \
    --output_dir sol_runs_h11_5_g2__cy7447__s42
```

Then, run `CPSAT-closure.py` on those generated solutions.

### Arguments for `CPSAT-closure.py`:

- `-h, --help`: show this help message and exit
- `--solutions`: Glob pattern for partial solutions files (e.g., `solutions_gpu_*.json` or `.jsonl`). (Required)
- `--geometry`: Path to the parsed geometry JSON file (e.g., `cy_geometry_exports/all_geometry_h11_5.json`). (Required)
- `--cy_index`: Index of the CY manifold in the database. (Required)
- `--gamma`: Target Gamma value used. (Required)
- `--time_limit`: Per-K0 CP-SAT solver wall-time cap in seconds. (default: 5)
- `--total_time_limit`: Whole-run wall-time budget across all files and K0s in seconds.
- `--max_matrices`: Max number of matrices to process from each input file. (default: 1000)
- `--stability_mode`: Strategy for Bogomolov stability constraints: `lazy`, `eager`, or `skip`. (default: lazy)
- `--workers`: Number of parallel workers for the CP-SAT solver. (default: 8)
- `--m_bound`: Max integer charge bound for matrix elements. (default: 8)
- `--tail_col_bound`: Bound on the last column of K (defaults to `--m_bound`).
- `--objective_cols`: Number of leading columns to include in the L1 objective.
- `--apply_column_lex`: Break column-permutation symmetry via lex ordering. (default: False)
- `--output_dir`: Directory to write the completed `closed_*.json` solutions. (Required)

**Example Usage**:
```bash
python CPSAT-closure.py \
    --solutions sol_runs_h11_5_g2__cy7447__s42/solutions_gpu_h11_5_idx_7447_*.jsonl \
    --geometry cy_geometry_exports/all_geometry_h11_5.json \
    --cy_index 7447 \
    --gamma 2 \
    --time_limit 5 \
    --max_matrices 1000 \
    --stability_mode lazy \
    --workers 8 \
    --output_dir closed_runs_h11_5_g2__cy7447
```

## Post-Processing Scripts

All evaluation and filtering scripts located in `scripts/` are designed to be run standalone and have an explicit CLI interface with sensible defaults.

### 1. `scripts/create_LB-Explorer_inputs.py`
Generates geometry input files from the full CICY database for the RL agent to use.
- `--db_path`: Path to input full_cicy_database.json (default: `databases/full_cicy_database.json`)
- `--output_dir`: Directory to save generated geometry inputs (default: `cy_geometry_exports`)

**Example Usage**:
```bash
python scripts/create_LB-Explorer_inputs.py \
    --db_path databases/full_cicy_database.json \
    --output_dir cy_geometry_exports
```

### 2. `scripts/check_validity_solutions.py`
Standalone Solution Verifier for Line Bundle Solutions. Evaluates candidate integer matrices against physics constraints.
- `--rank`: Rank of the vector bundle (default: 5)
- `--m_bound`: Max integer charge bound for matrices (default: 5)
- `--workers`: Number of worker processes to use (default: 8)
- `--db_path`: Path to the CICY database JSON file (default: `databases/full_cicy_database.json`)
- `--input_dir`: Directory containing the raw_cy_*.jsonl files to verify (default: `Sol_Runs`). Use `Sol_Runs_TL` for transfer learning outputs.

**Example Usage**:
```bash
python scripts/check_validity_solutions.py \
    --rank 5 \
    --workers 4 \
    --input_dir Sol_Runs
```

### 3. `scripts/check_equivariance.py`
Check equivariance of line bundle solutions under freely acting symmetries.
- `--workers`: Number of worker processes to use (default: 8)
- `--cy_index`: List of CY IDs to process (default: all found in input_dir)
- `--db_path`: Path to the CICY database JSON file (default: `databases/full_cicy_database.json`)
- `--input_dir`: Directory containing raw_cy_*.jsonl files (default: `Sol_Runs`)
- `--output_csv`: Output path for CSV stats (default: `Analysis_Plots/equivariance_stats.csv`)

**Example Usage**:
```bash
python scripts/check_equivariance.py \
    --cy_index 7890 \
    --workers 8
```

### 4. `scripts/check_polystability.py`
Check exact polystability of line bundle matrices by verifying existence of Kahler parameters.
- `files`: Optional glob pattern(s) or file paths for raw_*.jsonl files (overrides input_dir)
- `--workers`: Number of parallel worker processes (default: 8)
- `--db_path`: Path to CICY database JSON file (default: `databases/full_cicy_database.json`)
- `--input_dir`: Folder containing raw_*.jsonl solutions to check (default: `Sol_Runs`)

**Example Usage**:
```bash
python scripts/check_polystability.py \
    --workers 4
```
> [!WARNING]
> **Reliability of Polystability Checks:**
> This script uses a PyTorch-based pre-filter (evaluating a fixed number of Sobol sequence points) followed by a local SciPy optimizer (SLSQP) to find Kahler parameters that satisfy the polystability conditions.
> - **Acceptance (High Trust):** If the script finds a solution and marks a matrix as **stable**, you can trust the result. 
> - **Rejection (Low Trust for large $h^{1,1}$):** If the script rejects a matrix (fails to find Kahler parameters), it may be a **false negative**. The pre-filter uses a fixed threshold (`< 1.0`) and the Sobol sequence is very sparse for large $h^{1,1}$. For $h^{1,1} \ge 6$ (which makes up ~81% of the database) or for large intersection numbers, the script will almost always fail to find the global minimum and incorrectly reject valid solutions. Proceed with caution when interpreting negative results. In [arXiv:2607.00078](https://arxiv.org/abs/2607.00078) and scripts/check_validity_solutions.py, the polystability is checked by requiring that the matrix $(M_a)_{jk} = \kappa_{ijk}\mathbf{k}^i_a$ has at least one positive and one negative entry. Moreover, the same should hold for every linear combination $v^aM_a$, with $v_a$ being a vector with integer entries between $-2$ and $2$ (see e.g. [arXiv:2306.03147](https://arxiv.org/abs/2306.03147) for a similar check of the polystability condition).

### 5. `scripts/check_lb_spec.py`
Check exact spectrum and equivariance constraints for line bundle solutions.
- `--m_bound`: Max charge bound for matrix elements (default: 2)
- `--workers`: Number of parallel workers (default: 8)
- `--cy_index`: List of CY IDs to process (default: all)
- `--gamma`: Optional: process only this gamma value
- `--only_trivial`: Only process manifolds with trivial equivariance (default: False)
- `--db_path`: Path to CICY database JSON file (default: `databases/full_cicy_database.json`)
- `--input_dir`: Folder containing raw_cy_*.jsonl solutions to check (default: `Sol_Runs`)
- `--output_dir`: Folder to save spectrum CSVs (default: `Analysis_Plots/Spectrum`)

**Example Usage**:
```bash
python scripts/check_lb_spec.py \
    --m_bound 2 \
    --cy_index 7890
```

### 6. `scripts/generate_sym_groups.py`
Generate topological and configuration symmetry groups for CICYs. NOTE: Running this script requires SageMath to be installed on your system.
- `--db_path`: Path to input CICY database JSON (default: `databases/full_cicy_database.json`)
- `--output_path`: Path to save output JSON with symmetries (default: `databases/cicy_symmetries.json`)

**Example Usage**:
```bash
sage -python scripts/generate_sym_groups.py \
    --db_path databases/full_cicy_database.json
```

