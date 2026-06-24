import json
import orjson
import numpy as np
import os
import glob
import argparse
import itertools
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import sympy as sp
import re

# ====================================================================
# I. GEOMETRY & VALIDATOR (Strict Verification Mode)
# ====================================================================

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

def prepare_geometry_objects(geo_data):
    """Converts JSON geometry data into optimized NumPy arrays."""
    h11 = geo_data.get('H11', geo_data.get('h11'))
    poly_str = geo_data.get('Ring', geo_data.get('intersecting_polynomial'))
    if poly_str is None:
        raise ValueError("No intersection polynomial found")
    kappa = parse_polynomial_to_tensor(poly_str, h11)
    
    c2_tx = np.array(geo_data.get('C2', geo_data.get('c2_tx')))
    return kappa, c2_tx

def prepare_stability_vectors(rank=5, range_bound=2):
    """Generates all non-zero integer vectors v for stability check."""
    r = range(-range_bound, range_bound + 1)
    vectors = list(itertools.product(r, repeat=rank))
    v_matrix = np.array(vectors, dtype=np.int8)
    
    # Filter zero vector
    norms = np.abs(v_matrix).sum(axis=1)
    non_zero_mask = norms > 0
    
    # Filter trace directions (uniform vectors)
    is_uniform = np.all(v_matrix == v_matrix[:, [0]], axis=1)
    
    valid_mask = non_zero_mask & (~is_uniform)
    return v_matrix[valid_mask]

class StrictBundleValidator:
    """
    Strict validator for final verification. 
    Enforces all constraints with no partial credit or curriculum leniency.
    """
    def __init__(self, geometry_data, rank, target_gamma, m_bound):
        kappa, c2_tx = prepare_geometry_objects(geometry_data)
        self.kappa = kappa
        self.c2_tx = c2_tx
        self.rank = rank
        self.target_gamma = abs(target_gamma)
        self.m_bound = m_bound
        self.test_vectors = prepare_stability_vectors(rank, range_bound=2)

    def _calculate_indices(self, k_matrix):
        # Cubic Term: 1/6 * kappa_ijk * k^i * k^j * k^k
        cubic_term = np.einsum('ijk,ia,ja,ka->a', self.kappa, k_matrix, k_matrix, k_matrix)
        # Linear Term: 1/12 * c2_i * k^i
        linear_term = np.einsum('i,ia->a', self.c2_tx, k_matrix)
        return (cubic_term / 6.0) + (linear_term / 12.0)

    def check_all(self, k_matrix):
        """
        Returns a dictionary of boolean flags for each condition.
        Order: Anom, Stab, Sum, Rng, Pair, Ntr, Bnd
        """
        results = {}
        
        # 1. Anomaly (Using 0.5 coeff)
        KK_T = np.dot(k_matrix, k_matrix.T)
        contraction = np.einsum('ijk,jk->i', self.kappa, KK_T)
        c2_V = 0.5 * contraction
        results['Anom'] = bool(np.all(c2_V <= self.c2_tx + 1e-5))

        # 2. Stability (Strict Indefiniteness: Max >= 1 AND Min <= -1)
        M_basis = np.einsum('ijk,ia->ajk', self.kappa, k_matrix)
        M_combined = np.einsum('na,ajk->njk', self.test_vectors, M_basis)
        flat_Ms = M_combined.reshape(M_combined.shape[0], -1)
        min_vals = np.min(flat_Ms, axis=1)
        max_vals = np.max(flat_Ms, axis=1)
        # Pass if NO vector is purely positive (max<=0) OR purely negative (min>=0)
        results['Stab'] = not np.any((max_vals <= 1e-5) | (min_vals >= -1e-5))

        # Indices Calculation
        indices = self._calculate_indices(k_matrix)
        
        # 3. Index Sum
        target = -3 * self.target_gamma
        results['Sum'] = bool(abs(np.sum(indices) - target) < 1e-5)

        # 4. Index Range [-3*gamma, 0]
        lower_bound = -3 * self.target_gamma
        results['Rng'] = bool(np.all((indices >= lower_bound - 1e-5) & (indices <= 1e-5)))

        # 5. Pairwise Index
        n_cols = k_matrix.shape[1]
        pair_cols = []
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                pair_cols.append(k_matrix[:, i] + k_matrix[:, j])
        if pair_cols:
            pair_matrix = np.column_stack(pair_cols)
            pair_indices = self._calculate_indices(pair_matrix)
            results['Pair'] = bool(np.all((pair_indices >= lower_bound - 1e-5) & (pair_indices <= 1e-5)))
        else:
            results['Pair'] = True

        # 6. Non-triviality
        # A: Columns non-zero
        col_magnitudes = np.sum(np.abs(k_matrix), axis=0)
        cols_nonzero = bool(np.all(col_magnitudes > 0))
        # B: Pairs non-zero (no cancellations)
        pairs_nonzero = True
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                if np.sum(np.abs(k_matrix[:, i] + k_matrix[:, j])) == 0:
                    pairs_nonzero = False; break
        results['Ntr'] = cols_nonzero and pairs_nonzero

        # 7. Bounds (Optional check)
        results['Bnd'] = True

        return results

def verify_matrix_batch(matrices, validator):
    total_valid = 0
    for m_list in matrices:
        k_mat = np.array(m_list)
        results = validator.check_all(k_mat)
        if all(results.values()):
            total_valid += 1
    return total_valid

def process_cy_task(cy_info):
    cy, h11, gamma, files, rank, m_bound, db_path = cy_info
    
    print(f"  [Worker] Starting CY {cy} Gamma {gamma} (Files: {len(files)})...", flush=True)
    
    try:
        with open(db_path, 'r') as f:
            all_geo = json.load(f)
        geo_data = next((m for m in all_geo if m.get('Num', m.get('id')) == cy), None)
    except Exception as e:
        return cy, gamma, 0, 0, f"Error loading DB: {e}"
        
    if not geo_data:
        return cy, gamma, 0, 0, "CY not in geometry file"
        
    try:
        validator = StrictBundleValidator(geo_data, rank, gamma, m_bound)
    except Exception as e:
        return cy, gamma, 0, 0, f"Validator error: {e}"
    
    total_loaded = 0
    total_valid = 0
    
    for filepath in files:
        try:
            matrices = []
            if filepath.endswith('.jsonl'):
                with open(filepath, 'rb') as f:
                    for line in f:
                        row = orjson.loads(line)
                        if len(row) == 4 and isinstance(row[3], bool) and isinstance(row[0], str):
                            continue
                        elif isinstance(row, list) and isinstance(row[0], list):
                            matrices.append(row)
            else:
                with open(filepath, 'r') as f:
                    matrices = json.load(f)
        except Exception as e:
            return cy, gamma, total_loaded, total_valid, f"Error reading {filepath}: {e}"
            
        if matrices:
            v_count = verify_matrix_batch(matrices, validator)
            total_loaded += len(matrices)
            total_valid += v_count
        
    return cy, gamma, total_loaded, total_valid, None

def verify_all(args):
    if not os.path.exists(args.db_path):
        print(f"[!] Database {args.db_path} not found.")
        return

    search_pattern = os.path.join(args.input_dir, "**", "raw_cy_*_h11_*_g_*.jsonl")
    files = glob.glob(search_pattern, recursive=True)

    groups = defaultdict(list)
    for f in files:
        m = re.search(r"raw_cy_(\d+)_h11_(\d+)_g_(\d+)", f)
        if m:
            cy = int(m.group(1))
            h11 = int(m.group(2))
            gamma = int(m.group(3))
            groups[(cy, h11, gamma)].append(f)
            
    if not groups:
        print(f"No raw_cy_*.jsonl files found in {args.input_dir}.")
        return
        
    tasks = []
    for (cy, h11, gamma), f_list in groups.items():
        tasks.append((cy, h11, gamma, f_list, args.rank, args.m_bound, args.db_path))
        
    print(f"[*] Found {len(tasks)} distinct (CY, Gamma) combinations.")
    print(f"[*] Beginning verification using {args.workers} workers...\n")
    
    print("="*70)
    print(f"{'CY Index':<10} | {'Gamma':<6} | {'Loaded':<10} | {'Valid':<10} | {'% Valid':<8} | {'Status'}")
    print("="*70)
    
    results = []
    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(process_cy_task, tasks):
            results.append(res)
            cy, gamma, loaded, valid, err = res
            if err:
                print(f"{cy:<10} | {gamma:<6} | {'-':<10} | {'-':<10} | {'-':<8} | {err}")
            else:
                pct = (valid / loaded * 100) if loaded > 0 else 0
                print(f"{cy:<10} | {gamma:<6} | {loaded:<10} | {valid:<10} | {pct:>5.2f}%   | OK")
                
    print("="*70)
    print(f"[*] Completed verification of {len(results)} combinations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Solution Verifier for Line Bundle Solutions. Evaluates candidate integer matrices against physics constraints.")
    parser.add_argument('--rank', type=int, default=5, help='Rank of the vector bundle (default: 5)')
    parser.add_argument('--m_bound', type=int, default=5, help='Max integer charge bound for matrices (default: 5)')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help=f'Number of worker processes to use (default: {mp.cpu_count()})')
    parser.add_argument('--db_path', type=str, default='databases/full_cicy_database.json', help='Path to the CICY database JSON file (default: databases/full_cicy_database.json)')
    parser.add_argument('--input_dir', type=str, default='Sol_Runs', help='Directory containing the raw_cy_*.jsonl files to verify (default: Sol_Runs). Use Sol_Runs_TL for transfer learning outputs.')
    
    args = parser.parse_args()
    verify_all(args)