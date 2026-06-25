#!/usr/bin/env python3
"""
check_exact_polystability.py
============================
Checks exact poly-stability for line bundle matrices by verifying if there
exists a Kahler parameter vector t > 0 such that sum_a (t^T M_a t)^2 = 0.
Assumes c_1(V) = 0 for all evaluated matrices.
"""

import os
import sys
import glob
import json
import orjson
import numpy as np
import sympy as sp
import re
import argparse
import hashlib
import multiprocessing as mp
from functools import partial
import torch
from tqdm import tqdm
from collections import defaultdict
from check_equivariance import build_symmetry_group
from scipy.optimize import minimize, linprog


def parse_polynomial_to_tensor(poly_str, h11):
    """Parses a sympy intersection polynomial string into a 3D numpy tensor kappa."""
    if h11 == 0:
        return np.zeros((0, 0, 0))

    s = re.sub(r"J\((\d+)\)", r"J_\1", poly_str)
    s = s.replace("^", "**")

    if not s.strip() or s.strip() == "0":
        return np.zeros((h11, h11, h11))

    # Sympy symbols handles 1 or multiple automatically as a tuple when using colon
    J_vars = sp.symbols(f"J_1:{h11+1}")

    expr = sp.expand(sp.sympify(s))
    kappa = np.zeros((h11, h11, h11))

    for i in range(h11):
        for j in range(i, h11):
            for k in range(j, h11):
                term = J_vars[i] * J_vars[j] * J_vars[k]
                c = expr.coeff(term)
                if c != 0:
                    val = float(c)

                    # Symmetrize
                    kappa[i, j, k] = val
                    kappa[i, k, j] = val
                    kappa[j, i, k] = val
                    kappa[j, k, i] = val
                    kappa[k, i, j] = val
                    kappa[k, j, i] = val

    return kappa


def run_slsqp_worker(args):
    idx, k_matrix, t0, kappa, h11 = args
    M_matrices = np.einsum("ijk,ia->ajk", kappa, k_matrix)

    def objective(t):
        M_t = np.dot(M_matrices, t)
        t_M_t = np.dot(M_t, t)
        return np.sum(t_M_t**2)

    def jacobian(t):
        M_t = np.dot(M_matrices, t)
        t_M_t = np.dot(M_t, t)
        return 4.0 * np.dot(t_M_t, M_t)

    constraints = [{"type": "eq", "fun": lambda t: np.sum(t) - 1.0}]
    bounds = [(0, None) for _ in range(h11)]

    res = minimize(
        objective,
        t0,
        method="SLSQP",
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
        tol=1e-12,
    )
    is_stable = res.fun < 1e-10 and np.all(res.x > 1e-6)
    return idx, is_stable


def run_hybrid_pytorch_scipy(unique_matrices, kappa, h11, workers=1, seed=42):
    # Guarantee mathematical determinism
    torch.manual_seed(seed)

    N = len(unique_matrices)
    if N == 0:
        return np.zeros(0, dtype=bool)

    unique_matrices_np = np.array(unique_matrices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    kappa_t = torch.tensor(kappa, dtype=dtype, device=device)
    K_t = torch.tensor(unique_matrices_np, dtype=dtype, device=device)

    total_samples = 10000
    matrix_chunk_size = 5000

    best_loss = torch.full((N,), float("inf"), dtype=dtype, device=device)
    best_t = torch.zeros((N, h11), dtype=dtype, device=device)

    print(
        f"  [*] Pre-generating {total_samples} low-discrepancy Sobol sequence points..."
    )
    soboleng = torch.quasirandom.SobolEngine(dimension=h11, scramble=True, seed=seed)
    all_t_samples = soboleng.draw(total_samples).to(dtype=dtype, device=device)
    all_t_samples /= torch.sum(all_t_samples, dim=-1, keepdim=True)

    # Mathematical optimization: Precompute T_si = sum_{j,k} kappa_ijk t_sj t_sk
    # This avoids constructing the gigantic M_t tensor for every matrix, saving ~20GB of RAM.
    print(
        f"  [*] Precomputing topological intersection constants across all {total_samples} samples..."
    )
    T_s = torch.einsum("ijk,sj,sk->si", kappa_t, all_t_samples, all_t_samples)

    num_matrix_chunks = (N + matrix_chunk_size - 1) // matrix_chunk_size
    print(
        f"  [*] Evaluating initializations across {N} matrices in {num_matrix_chunks} chunks (RAM safe)..."
    )

    for m_start in range(0, N, matrix_chunk_size):
        m_end = min(m_start + matrix_chunk_size, N)
        current_N = m_end - m_start

        K_chunk = K_t[m_start:m_end]

        # t_M_t_sna = sum_i T_si * K_nia
        t_M_t = torch.einsum("si,nia->sna", T_s, K_chunk)
        loss = torch.sum(t_M_t**2, dim=-1)

        s_min_loss, s_min_idx = torch.min(loss, dim=0)

        best_loss[m_start:m_end] = s_min_loss
        best_t[m_start:m_end] = all_t_samples[s_min_idx, :]

        if num_matrix_chunks > 1 and (m_start // matrix_chunk_size + 1) % 10 == 0:
            print(
                f"      - Processed chunk {m_start//matrix_chunk_size + 1}/{num_matrix_chunks}"
            )

    best_t_np = best_t.cpu().numpy()

    # Pre-filter: if best initialization didn't even reach 1.0, it's virtually impossible to be a root
    promising_local_indices = (
        torch.nonzero(best_loss < 1.0, as_tuple=True)[0].cpu().numpy()
    )

    print(
        f"  [*] Handoff: Running SciPy SLSQP precisely on {len(promising_local_indices)} promising matrices using {workers} workers..."
    )

    tasks = [
        (i, unique_matrices_np[i], best_t_np[i], kappa, h11)
        for i in promising_local_indices
    ]

    global_stable = np.zeros(N, dtype=bool)
    if tasks:
        with mp.Pool(workers) as pool:
            for original_idx, is_stable in tqdm(
                pool.imap_unordered(run_slsqp_worker, tasks, chunksize=100),
                total=len(tasks),
                desc="  SciPy SLSQP",
                unit="matrix",
            ):
                global_stable[original_idx] = is_stable

    return global_stable


def process_file(filepath, db, workers):
    basename = os.path.basename(filepath)
    out_dir = os.path.dirname(filepath)
    out_name = "exact_" + basename
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        print(f"[*] Skipping {basename} as output already exists.")
        return

    m1 = re.search(r"raw_cy_(\d+)_h11_(\d+)", basename)
    m2 = re.search(r"__cy(\d+)__", basename)
    h_match = re.search(r"h11_(\d+)", basename)

    if m1:
        cy_id = int(m1.group(1))
        h11 = int(m1.group(2))
    elif m2 and h_match:
        cy_id = int(m2.group(1))
        h11 = int(h_match.group(1))
    else:
        print(f"[!] Could not extract CY ID and h11 from filename: {basename}")
        return

    cy_data = next(
        (item for item in db if item.get("Num", item.get("id")) == cy_id), None
    )
    if not cy_data:
        print(f"[!] CY {cy_id} not found in database. Skipping {basename}.")
        return

    poly_str = cy_data.get("Ring", cy_data.get("intersecting_polynomial"))
    if not poly_str:
        print(f"[!] Intersection polynomial not found for CY {cy_id}. Skipping.")
        return

    kappa = parse_polynomial_to_tensor(poly_str, h11)

    matrices = []
    print(f"[*] Reading matrices from {basename}...")
    try:
        with open(filepath, "rb") as f:
            for line in f:
                if line.strip():
                    matrices.append(orjson.loads(line))
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
        return

    if not matrices:
        print(f"[*] No matrices found in {basename}.")
        return

    # Symmetry-aware canonicalisation to reduce matrix evaluations
    gSym, _ = build_symmetry_group(cy_id, db)
    if not gSym:
        gSym = [list(range(h11))]

    canonical_to_originals = defaultdict(list)
    print(f"[*] Canonicalising {len(matrices)} matrices...")
    for mat in matrices:
        best_full = None
        for p in gSym:
            # Permute rows
            permuted_rows = [mat[p[i]] for i in range(len(p))]
            # Full canonical form: sort columns
            full_key = tuple(sorted(zip(*permuted_rows)))
            if best_full is None or full_key < best_full:
                best_full = full_key
        canonical_to_originals[best_full].append(mat)

    unique_matrices = [list(zip(*k)) for k in canonical_to_originals.keys()]

    print(
        f"[*] Reduced from {len(matrices)} to {len(unique_matrices)} unique canonical matrices."
    )
    print(
        f"[*] Checking exact polystability for {len(unique_matrices)} canonical matrices..."
    )

    print(f"[*] Running Hybrid PyTorch-SciPy exact root finder...")
    stable_mask = run_hybrid_pytorch_scipy(unique_matrices, kappa, h11, workers=workers)

    stable_matrices = []
    for is_stable, canon_mat in zip(stable_mask, unique_matrices):
        if is_stable:
            canon_key = tuple(zip(*canon_mat))
            stable_matrices.extend(canonical_to_originals[canon_key])

    with open(out_path, "wb") as f:
        for mat in stable_matrices:
            f.write(orjson.dumps(mat) + b"\n")

    print(
        f"[+] Found {len(stable_matrices)} stable solutions ({(len(stable_matrices)/len(matrices))*100:.2f}%)."
    )
    print(f"[+] Saved stable solutions to: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check exact polystability of line bundle matrices by verifying existence of Kahler parameters."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=[],
        help="Optional glob pattern(s) or file paths for raw_*.jsonl files (overrides input_dir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel worker processes (default: {mp.cpu_count()})",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="databases/full_cicy_database.json",
        help="Path to CICY database JSON file (default: databases/full_cicy_database.json)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Sol_Runs",
        help="Folder containing raw_*.jsonl solutions to check (default: Sol_Runs)",
    )
    args = parser.parse_args()

    filepaths = []
    if args.files:
        for pattern in args.files:
            expanded = glob.glob(pattern)
            if expanded:
                filepaths.extend(expanded)
            else:
                filepaths.append(pattern)
    else:
        in_dir = args.input_dir
        filepaths = glob.glob(
            os.path.join(in_dir, "**", "raw_cy_*.jsonl"), recursive=True
        )

    # Sort the list to guarantee deterministic processing order
    filepaths = sorted(list(set(filepaths)))

    if not filepaths:
        print("No files found to process.")
        return

    print(f"Loading database from {args.db_path}...")
    if not os.path.exists(args.db_path):
        print(f"CRITICAL ERROR: Database {args.db_path} not found.")
        return

    with open(args.db_path, "r") as f:
        db = json.load(f)

    print(f"Processing {len(filepaths)} files with {args.workers} workers...\n")

    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"[!] File not found: {fp}")
            continue
        process_file(fp, db, args.workers)


if __name__ == "__main__":
    main()
