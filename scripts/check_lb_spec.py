#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import argparse
import glob
import re
import time
import signal
import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sympy.combinatorics import Permutation, PermutationGroup
import pyCICY
import gc
import hashlib
import orjson


# Fix for AttributeError: module 'numpy' has no attribute 'int' in pyCICY
if not hasattr(np, "int"):
    np.int = int

# Fix for ValueError: setting an array element with a sequence (inhomogeneous shape)
_old_array = np.array


def _new_array(*args, **kwargs):
    try:
        return _old_array(*args, **kwargs)
    except ValueError as e:
        if "inhomogeneous shape" in str(
            e
        ) or "setting an array element with a sequence" in str(e):
            return _old_array(*args, **kwargs, dtype=object)
        raise e


np.array = _new_array


# --- Euler Violation Handler ---
class EulerViolationHandler:
    def __init__(self):
        self.euler_violated = False

    def handle_warning(self, message, category, filename, lineno, file=None, line=None):
        if "Euler violated" in str(message):
            self.euler_violated = True


_euler_handler = EulerViolationHandler()
warnings.showwarning = _euler_handler.handle_warning


# --- Symmetry Helpers ---
def stable_hash(obj):
    """Compact, stable binary hash of a Python object."""
    return hashlib.md5(repr(obj).encode()).digest()


def build_symmetry_group(cy_id, db):
    cy_entry = next((e for e in db if e["id"] == cy_id), None)
    if not cy_entry:
        return None, None
    h11 = cy_entry.get("h11")
    raw_gens = cy_entry.get("sym_gen", [])
    if not raw_gens:
        gSym = [list(range(h11))]
    else:
        zero_based_gens = []
        for cycle_list in raw_gens:
            zero_based_cycles = [[x - 1 for x in cycle] for cycle in cycle_list]
            zero_based_gens.append(Permutation(zero_based_cycles, size=h11))
        group = PermutationGroup(zero_based_gens)
        gSym = [[p(i) for i in range(h11)] for p in group.generate()]
    return gSym, h11


def canonicalise(V, gSym):
    """
    Returns (col_key, geom_key, full_key).
    V is 5 x h11.
    """
    # 1. Col: Polynomial only (sort the 5 line bundle rows)
    col_key = tuple(sorted(tuple(row) for row in V))

    if not gSym:
        # Fallback if no geometric symmetry
        raw_key = tuple(tuple(row) for row in V)
        return col_key, raw_key, col_key

    best_geom = None
    best_full = None

    for p in gSym:
        # Permute ambient space columns
        permuted_V = [[row[i] for i in p] for row in V]

        # Geom key: Column permutation only (ambient space)
        geom_key = tuple(tuple(row) for row in permuted_V)
        # Full key: Column permutation + Row sorting (ambient + polynomial)
        full_key = tuple(sorted(tuple(row) for row in permuted_V))

        if best_geom is None or geom_key < best_geom:
            best_geom = geom_key
        if best_full is None or full_key < best_full:
            best_full = full_key

    return col_key, best_geom, best_full


# --- Worker Functions ---
def scan_and_canonicalise_worker(args):
    """Worker to read a chunk of the JSONL file, canonicalise, and return hashes."""
    file_path, start_line, end_line, gSym, kmax = args
    
    events = [] # (h_c, h_g, h_f, in_range)
    representative_matrices = {} # h_c -> matrix (only if in_range)
    
    try:
        with open(file_path, "rb") as f:
            for i, line in enumerate(f):
                if i < start_line: continue
                if i >= end_line: break
                
                m_raw = orjson.loads(line)
                # Convert to 5 x h11
                m_np = np.array(m_raw).T
                m_list = m_np.tolist()
                
                # Canonicalise
                ck, gk, fk = canonicalise(m_list, gSym)
                h_c, h_g, h_f = stable_hash(ck), stable_hash(gk), stable_hash(fk)
                
                # Check kmax
                in_range = all(all(abs(k) <= kmax for k in row) for row in m_list)
                
                events.append((h_c, h_g, h_f, in_range))
                if in_range and h_c not in representative_matrices:
                    representative_matrices[h_c] = m_list
                    
    except Exception as e:
        print(f"Error in worker: {e}")
        
    return events, representative_matrices


# --- Cohomology Workers ---
_worker_X = None
_worker_mode = None
_worker_divisor_group = None

def get_divisor_action_info(cy_entry, g):
    fas = cy_entry.get("Mapped Freely Acting Symmetries", [])
    matching = [s for s in fas if s[1] == g]

    if not matching:
        print(f"  Warning: No Mapped Freely Acting Symmetry with order {g} found.")
        return "trivial", None

    for s in matching:
        if s[2] == [] or s[2] is None:
            return "trivial", None

    perm_generators_raw = matching[0][2]
    h11 = cy_entry.get("H11")

    generators = []
    for cycle_list in perm_generators_raw:
        perm = list(range(h11))
        for cycle in cycle_list:
            zero_cycle = [x - 1 for x in cycle]
            for i in range(len(zero_cycle)):
                perm[zero_cycle[i]] = zero_cycle[(i + 1) % len(zero_cycle)]
        generators.append(perm)

    divisor_group = generate_full_group(generators, h11)
    return "nontrivial", divisor_group

def generate_full_group(generators, h11):
    identity = list(range(h11))
    def compose(p1, p2):
        return [p1[p2[i]] for i in range(h11)]
    def perm_key(p):
        return tuple(p)
    group = {perm_key(identity): identity}
    for gen in generators:
        k = perm_key(gen)
        if k not in group:
            group[k] = gen
    changed = True
    while changed:
        changed = False
        current_elements = list(group.values())
        for elem in current_elements:
            for gen in generators:
                for new in [compose(elem, gen), compose(gen, elem)]:
                    k = perm_key(new)
                    if k not in group:
                        group[k] = new
                        changed = True
    return list(group.values())

def _check_equivariance_nontrivial(line_bundles, gamma):
    global _worker_X, _worker_divisor_group
    h11 = len(line_bundles[0])
    bundles = [tuple(int(x) for x in L) for L in line_bundles]

    bundle_counts = {}
    for b in bundles:
        bundle_counts[b] = bundle_counts.get(b, 0) + 1

    chi_cache = {}
    for b in set(bundles):
        try:
            chi_cache[b] = int(_worker_X.line_co_euler(list(b)))
        except:
            return False

    processed_bundles = set()
    for b in set(bundles):
        if b in processed_bundles:
            continue

        orbit_val = set()
        for perm in _worker_divisor_group:
            permuted = tuple(b[perm[j]] for j in range(h11))
            orbit_val.add(permuted)

            if permuted not in bundle_counts:
                return False
            if chi_cache[permuted] != chi_cache[b]:
                return False

        count = bundle_counts[b]
        for ob in orbit_val:
            if bundle_counts[ob] != count:
                return False
            processed_bundles.add(ob)

        orbit_chi = sum(chi_cache[ob] for ob in orbit_val) * count
        if (orbit_chi % gamma) != 0:
            return False

    return True


def init_worker(cicy_conf_with_dims, mode, divisor_group):
    """Initializes the CICY manifold once per worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_X, _worker_mode, _worker_divisor_group
    _worker_mode = mode
    _worker_divisor_group = divisor_group
    # Import inside to ensure monkeypatches are active if needed
    import texttable

    original_init = texttable.Texttable.__init__

    def new_init(self, max_width=0):
        original_init(self, max_width=0)

    texttable.Texttable.__init__ = new_init

    from pyCICY import CICY

    _worker_X = CICY(cicy_conf_with_dims)


def check_spectrum(item):
    """Processes a single matrix and returns its cohomologies and pass/fail status."""
    line_bundles, gamma = item
    global _worker_X, _worker_mode

    h1_V, h2_V, chi_V = 0, 0, 0
    h1_W, h2_W, chi_W = 0, 0, 0

    # Track unique line bundles and their multiplicities for cond6
    unique_line_bundles = {}  # tuple(L) -> {'multiplicity': m, 'chi': chi}

    for L in line_bundles:
        L_tuple = tuple(int(x) for x in L)
        if L_tuple not in unique_line_bundles:
            try:
                cohomology = _worker_X.line_co(list(L_tuple))
            except:
                return "error"
            chi_L = int(_worker_X.line_co_euler(list(L_tuple)))
            unique_line_bundles[L_tuple] = {
                "multiplicity": 1,
                "chi": chi_L,
                "h1": int(cohomology[1]),
                "h2": int(cohomology[2]),
            }
        else:
            unique_line_bundles[L_tuple]["multiplicity"] += 1

    # Aggregate V totals from unique bundles
    for info in unique_line_bundles.values():
        m = info["multiplicity"]
        h1_V += m * info["h1"]
        h2_V += m * info["h2"]
        chi_V += m * info["chi"]

    # Computations for \wedge^2 V
    from itertools import combinations

    for Li, Lj in combinations(line_bundles, 2):
        L_ij = [int(x) for x in (np.array(Li) + np.array(Lj))]
        try:
            cohomology = _worker_X.line_co(L_ij)
        except:
            return "error"
        h1_W += int(cohomology[1])
        h2_W += int(cohomology[2])
        chi_W += int(_worker_X.line_co_euler(L_ij))

    # Spectrum Filtering conditions
    cond1 = h1_V == 3 * gamma
    cond2 = h2_V == 0
    cond3 = h1_W == 3 * gamma + h2_W
    cond4 = h2_W != 0
    cond5 = (chi_V == -3 * gamma) and (chi_W == -3 * gamma)

    # cond6
    if _worker_mode == "trivial":
        cond6 = all(
            (info["multiplicity"] * info["chi"]) % gamma == 0
            for info in unique_line_bundles.values()
        )
    else:
        cond6 = _check_equivariance_nontrivial(line_bundles, gamma)

    passed_equiv = cond6
    passed_spectrum = all([cond1, cond2, cond3, cond4, cond5, cond6])
    return (passed_equiv, passed_spectrum)


def process_manifold(cy, h11, g, kmax, db, n_workers, raw_path=None, out_csv_dir=None, only_trivial=False):
    print(f"\nProcessing CY #{cy}, h11={h11}, g={g} (kmax={kmax})...")

    if raw_path is None:
        raw_path = f"Sol_Runs/raw_cy_{cy}_h11_{h11}_g_{g}.jsonl"
        
    if not os.path.exists(raw_path):
        print(f"  Error: {raw_path} not found.")
        return None

    # Count lines to split work
    line_count = 0
    with open(raw_path, "rb") as f:
        for _ in f: line_count += 1
        
    if line_count == 0:
        print("  No matrices found in raw file.")
        return None

    gSym, _ = build_symmetry_group(cy, db)
    
    # 1. Parallel Scan and Canonicalisation
    print(f"  Scanning and canonicalising {line_count} matrices in parallel...")
    batch_size = max(1, line_count // n_workers)
    scan_args = []
    for i in range(0, line_count, batch_size):
        scan_args.append((raw_path, i, min(i + batch_size, line_count), gSym, kmax))
        
    all_events = [] # (h_c, h_g, h_f, in_range)
    global_representative_matrices = {} # h_c -> matrix
    
    with Pool(n_workers) as p:
        for events, reps in p.imap_unordered(scan_and_canonicalise_worker, scan_args):
            all_events.extend(events)
            for h_c, m in reps.items():
                if h_c not in global_representative_matrices:
                    global_representative_matrices[h_c] = m
    
    gc.collect()

    # 2. Extract Stats and Filter for Range
    n_total = len(all_events)
    n_unique_col = len(set(e[0] for e in all_events))
    n_unique_geom = len(set(e[1] for e in all_events))
    n_unique_full = len(set(e[2] for e in all_events))

    # Events in range
    events_r = [e for e in all_events if e[3]]
    n_total_in_range = len(events_r)
    
    if n_total_in_range == 0:
        print(f"  No matrices satisfy kmax={kmax}.")
        return {
            "cy": cy, "h11": h11, "g": g,
            "n_total": n_total, "n_unique_col": n_unique_col, "n_unique_geom": n_unique_geom, "n_unique_full": n_unique_full,
            "n_total_in_range": 0, "n_unique_col_in_range": 0, "n_unique_geom_in_range": 0, "n_unique_full_in_range": 0,
            "n_total_in_range_equiv": 0, "n_unique_col_in_range_equiv": 0, "n_unique_geom_in_range_equiv": 0, "n_unique_full_in_range_equiv": 0,
            "n_total_in_range_spectrum": 0, "n_unique_col_in_range_spectrum": 0, "n_unique_geom_in_range_spectrum": 0, "n_unique_full_in_range_spectrum": 0,
            "n_errors_total": 0, "n_errors_col": 0, "n_errors_geom": 0, "n_errors_full": 0,
        }

    n_c_r = len(set(e[0] for e in events_r))
    n_g_r = len(set(e[1] for e in events_r))
    n_f_r = len(set(e[2] for e in events_r))

    # 3. Parallel Cohomology on UNIQUE physical configurations (h_c)
    unique_h_c_in_range = sorted(list(set(e[0] for e in events_r)))
    matrices_to_check = [global_representative_matrices[h] for h in unique_h_c_in_range]

    print(f"  Running cohomology checks on {len(matrices_to_check)} unique physical configurations (out of {n_total_in_range} in-range)...")
    
    cicy_entry = next((item for item in db if item.get("Num", item.get("id")) == cy), None)
    if cicy_entry is None:
        print(f"  Error: CICY Num {cy} not found in database.")
        return None
        
    mode, divisor_group = get_divisor_action_info(cicy_entry, g)
    print(f"  Equivariance mode: {mode}")
    
    if only_trivial and mode != "trivial":
        print(f"  Skipping CY #{cy} because it requires nontrivial equivariance check.")
        return None
    
    cicy_conf = cicy_entry["Conf"]
    # Prepend a column of 1s (ambient space dimensions n=1 for P^1 factors)
    cicy_conf_with_dims = [[sum(row) - 1] + [int(x) for x in row] for row in cicy_conf]

    unique_results = []
    with Pool(n_workers, initializer=init_worker, initargs=(cicy_conf_with_dims, mode, divisor_group)) as p:
        tasks = [(m, g) for m in matrices_to_check]
        for res in tqdm(
            p.imap(check_spectrum, tasks),
            total=len(tasks),
            desc=f"  Cohomology (CY {cy}, g={g})",
            leave=False,
        ):
            unique_results.append(res)

    # Map results back to unique h_c
    h_c_to_res = {h: unique_results[i] for i, h in enumerate(unique_h_c_in_range)}
    
    # 4. Collect Stats
    def get_stats_for_level(level_idx):
        # unique_keys: hash -> {'equiv': False, 'spec': False, 'error': False}
        unique_keys = {}
        for e in events_r:
            h_level = e[level_idx]
            h_c = e[0]
            res = h_c_to_res[h_c]
            
            if h_level not in unique_keys:
                unique_keys[h_level] = {"equiv": False, "spec": False, "error": False}
            
            if res == "error":
                unique_keys[h_level]["error"] = True
            else:
                if res[0]: unique_keys[h_level]["equiv"] = True
                if res[1]: unique_keys[h_level]["spec"] = True
        
        n_equiv = sum(1 for v in unique_keys.values() if v["equiv"])
        n_spec = sum(1 for v in unique_keys.values() if v["spec"])
        n_err = sum(1 for v in unique_keys.values() if v["error"])
        return n_equiv, n_spec, n_err

    n_c_r_e, n_c_r_s, n_err_c = get_stats_for_level(0)
    n_g_r_e, n_g_r_s, n_err_g = get_stats_for_level(1)
    n_f_r_e, n_f_r_s, n_err_f = get_stats_for_level(2)

    # Raw stats (no uniqueness)
    n_total_in_range_equiv = sum(1 for e in events_r if h_c_to_res[e[0]] != "error" and h_c_to_res[e[0]][0])
    n_total_in_range_spectrum = sum(1 for e in events_r if h_c_to_res[e[0]] != "error" and h_c_to_res[e[0]][1])
    n_errors_total = sum(1 for e in events_r if h_c_to_res[e[0]] == "error")

    # Final cleanup of huge lists
    del all_events
    del events_r
    del global_representative_matrices
    gc.collect()

    res_row = {
        "cy": cy,
        "h11": h11,
        "g": g,
        "n_total": n_total,
        "n_unique_col": n_unique_col,
        "n_unique_geom": n_unique_geom,
        "n_unique_full": n_unique_full,
        "n_total_in_range": n_total_in_range,
        "n_unique_col_in_range": n_c_r,
        "n_unique_geom_in_range": n_g_r,
        "n_unique_full_in_range": n_f_r,
        "n_total_in_range_equiv": n_total_in_range_equiv,
        "n_unique_col_in_range_equiv": n_c_r_e,
        "n_unique_geom_in_range_equiv": n_g_r_e,
        "n_unique_full_in_range_equiv": n_f_r_e,
        "n_total_in_range_spectrum": n_total_in_range_spectrum,
        "n_unique_col_in_range_spectrum": n_c_r_s,
        "n_unique_geom_in_range_spectrum": n_g_r_s,
        "n_unique_full_in_range_spectrum": n_f_r_s,
        "n_errors_total": n_errors_total,
        "n_errors_col": n_err_c,
        "n_errors_geom": n_err_g,
        "n_errors_full": n_err_f,
    }

    if out_csv_dir is None:
        out_csv_dir = "Analysis_Plots/Spectrum"
    out_csv = os.path.join(out_csv_dir, f"histogram_stats_spec_cy_{cy}_h11_{h11}_g_{g}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([res_row]).to_csv(out_csv, index=False)
    print(f"  Saved results to {out_csv}")
    return res_row


def main():
    parser = argparse.ArgumentParser(description="Check exact spectrum and equivariance constraints for line bundle solutions.")
    parser.add_argument("--kmax", type=int, default=2, help="Max charge bound for matrix elements (default: 2)")
    parser.add_argument("--workers", type=int, default=cpu_count(), help=f"Number of parallel workers (default: {cpu_count()})")
    parser.add_argument("--cy", type=int, nargs="+", help="List of CY IDs to process (default: all)")
    parser.add_argument("--gamma", type=int, help="Optional: process only this gamma value")
    parser.add_argument("--only_trivial", action="store_true", help="Only process manifolds with trivial equivariance (default: False)")
    parser.add_argument("--db_path", type=str, default="databases/full_cicy_database.json", help="Path to CICY database JSON file (default: databases/full_cicy_database.json)")
    parser.add_argument("--input_dir", type=str, default="Sol_Runs", help="Folder containing raw_cy_*.jsonl solutions to check (default: Sol_Runs)")
    parser.add_argument("--output_dir", type=str, default="Analysis_Plots/Spectrum", help="Folder to save spectrum CSVs (default: Analysis_Plots/Spectrum)")
    args = parser.parse_args()

    with open(args.db_path, "r") as f:
        db = json.load(f)

    in_dir = args.input_dir
    out_csv_dir = args.output_dir

    if args.cy:
        for cy_id in args.cy:
            # Find manifold info from DB
            entry = next((e for e in db if e.get("Num", e.get("id")) == cy_id), None)
            if entry:
                h11 = entry.get("H11", entry.get("h11"))
                gammas = entry.get("gamma", [])
                if args.gamma:
                    if args.gamma in gammas:
                        gammas = [args.gamma]
                    else:
                        print(f"Warning: Gamma {args.gamma} not found for CY {cy_id} in DB.")
                        gammas = []

                for g in gammas:
                    f = os.path.join(in_dir, f"raw_cy_{cy_id}_h11_{h11}_g_{g}.jsonl")
                    if os.path.exists(f):
                        print(f"  Processing {f}...")
                        process_manifold(cy_id, h11, g, args.kmax, db, args.workers, raw_path=f, out_csv_dir=out_csv_dir, only_trivial=args.only_trivial)
                    else:
                        pass
            else:
                print(f"Warning: CY {cy_id} not found in DB.")
    else:
        # Process all raw files
        gamma_suffix = f"g_{args.gamma}" if args.gamma else "g_*"
        
        raw_files = glob.glob(os.path.join(in_dir, "**", f"raw_cy_*_h11_*_{gamma_suffix}.jsonl"), recursive=True)
        if args.cy:
            cy_set = set(args.cy)
            raw_files = [f for f in raw_files if int(re.search(r"cy_(\d+)", f).group(1)) in cy_set]
            
        for f in raw_files:
            m = re.search(r"raw_cy_(\d+)_h11_(\d+)_g_(\d+)", f)
            if m:
                process_manifold(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    args.kmax,
                    db,
                    args.workers,
                    raw_path=f,
                    out_csv_dir=out_csv_dir,
                    only_trivial=args.only_trivial,
                )


if __name__ == "__main__":
    main()
