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


# --- Symmetry Helpers ---
def stable_hash(obj):
    """Compact, stable binary hash of a Python object."""
    return hashlib.md5(repr(obj).encode()).digest()


def build_symmetry_group(cy_id, db):
    """Build the ambient space symmetry group for canonicalisation.
    Uses 'Group Generators' from full_cicy_database.json.
    """
    cy_entry = next((e for e in db if e["Num"] == cy_id), None)
    if not cy_entry:
        return None, None
    h11 = cy_entry.get("H11")
    raw_gens = cy_entry.get("Group Generators", [])
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


def get_divisor_action_info(cy_entry, g):
    """Determine equivariance check mode from Mapped Freely Acting Symmetries.

    Returns:
        (mode, divisor_group)
        mode = 'trivial'    -> divisor_group = None
        mode = 'nontrivial' -> divisor_group = list of 0-indexed full permutation
                               arrays (all group elements)
    """
    fas = cy_entry.get("Mapped Freely Acting Symmetries", [])
    matching = [s for s in fas if s[1] == g]

    if not matching:
        # No symmetry of this order found; fall back to trivial check
        print(f"  Warning: No Mapped Freely Acting Symmetry with order {g} found.")
        return "trivial", None

    # Priority: pick one with empty [] permutation (trivial action on divisors)
    for s in matching:
        if s[2] == [] or s[2] is None:
            return "trivial", None

    # All have non-empty permutations; use the first one
    perm_generators_raw = matching[0][2]
    h11 = cy_entry.get("H11")

    # Convert cycle notation (1-indexed) to full permutation arrays (0-indexed)
    generators = []
    for cycle_list in perm_generators_raw:
        # Start with identity
        perm = list(range(h11))
        for cycle in cycle_list:
            # cycle is a list like [3, 4] meaning 3->4->3 (1-indexed)
            zero_cycle = [x - 1 for x in cycle]
            for i in range(len(zero_cycle)):
                perm[zero_cycle[i]] = zero_cycle[(i + 1) % len(zero_cycle)]
        generators.append(perm)

    # Generate full group by closure
    divisor_group = generate_full_group(generators, h11)
    return "nontrivial", divisor_group


def generate_full_group(generators, h11):
    """Generate all group elements from a set of permutation generators.

    Each generator is a list of length h11 representing the full permutation
    (0-indexed). Returns a list of all distinct group elements including identity.
    """
    identity = list(range(h11))

    def compose(p1, p2):
        """Compose two permutations: result[i] = p1[p2[i]]."""
        return [p1[p2[i]] for i in range(h11)]

    def perm_key(p):
        return tuple(p)

    group = {perm_key(identity): identity}

    # Also add generators themselves
    for gen in generators:
        k = perm_key(gen)
        if k not in group:
            group[k] = gen

    # BFS closure
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


def canonicalise(V, gSym):
    """
    Returns (col_key, geom_key, full_key).
    V is 5 x h11.
    """
    # 1. Col: Polynomial only (sort the 5 line bundle rows)
    col_key = tuple(sorted(tuple(row) for row in V))

    if not gSym:
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
    file_path, start_line, end_line, gSym = args

    events = []  # (h_c, h_g, h_f)
    representative_matrices = {}  # h_c -> matrix

    try:
        with open(file_path, "rb") as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                if i >= end_line:
                    break

                m_raw = orjson.loads(line)
                # Convert to 5 x h11
                m_np = np.array(m_raw).T
                m_list = m_np.tolist()

                # Canonicalise
                ck, gk, fk = canonicalise(m_list, gSym)
                h_c, h_g, h_f = stable_hash(ck), stable_hash(gk), stable_hash(fk)

                events.append((h_c, h_g, h_f))
                if h_c not in representative_matrices:
                    representative_matrices[h_c] = m_list

    except Exception as e:
        print(f"Error in worker: {e}")

    return events, representative_matrices


# --- Cohomology Workers ---
_worker_X = None
_worker_mode = None
_worker_divisor_group = None


def init_worker(cicy_conf_with_dims, mode, divisor_group):
    """Initializes the CICY manifold once per worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_X, _worker_mode, _worker_divisor_group
    _worker_mode = mode
    _worker_divisor_group = divisor_group

    import texttable

    original_init = texttable.Texttable.__init__

    def new_init(self, max_width=0):
        original_init(self, max_width=0)

    texttable.Texttable.__init__ = new_init

    from pyCICY import CICY

    _worker_X = CICY(cicy_conf_with_dims)


def check_equivariance(item):
    """Processes a single matrix and returns True if equivariance check passes.

    Two modes:
    - 'trivial': For each distinct line bundle L in V, m(L)*chi(X,L) ≡ 0 mod gamma.
    - 'nontrivial': The 5 line bundles must partition into orbits under the divisor
      group action, and chi(X, sum_of_orbit) ≡ 0 mod gamma for each orbit.
    """
    line_bundles, gamma = item
    global _worker_X, _worker_mode, _worker_divisor_group

    if _worker_mode == "trivial":
        return _check_equivariance_trivial(line_bundles, gamma)
    else:
        return _check_equivariance_nontrivial(line_bundles, gamma)


def _check_equivariance_trivial(line_bundles, gamma):
    """Original check: m(L) * chi(X, L) ≡ 0 mod gamma for each distinct L."""
    global _worker_X

    unique_line_bundles = {}  # tuple(L) -> {'multiplicity': m, 'chi': chi}

    for L in line_bundles:
        L_tuple = tuple(int(x) for x in L)
        if L_tuple not in unique_line_bundles:
            try:
                chi_L = int(_worker_X.line_co_euler(list(L_tuple)))
                unique_line_bundles[L_tuple] = {
                    "multiplicity": 1,
                    "chi": chi_L,
                }
            except:
                return False  # Assume fail on error
        else:
            unique_line_bundles[L_tuple]["multiplicity"] += 1

    cond6 = all(
        (info["multiplicity"] * info["chi"]) % gamma == 0
        for info in unique_line_bundles.values()
    )
    return cond6


def _check_equivariance_nontrivial(line_bundles, gamma):
    """Partition-based check with group action on divisors (columns).

    Follows the stricter criteria where EACH orbit must descend to the
    quotient manifold independently.
    1. Verify multiset invariance (M=0).
    2. Verify topological invariance (chi is constant on orbits).
    3. Verify that the total Euler characteristic of EACH orbit block
       is divisible by the group order.
    """
    global _worker_X, _worker_divisor_group
    h11 = len(line_bundles[0])
    bundles = [tuple(int(x) for x in L) for L in line_bundles]

    # 1. Count multiplicities
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

        # Compute orbit in value space
        orbit_val = set()
        for perm in _worker_divisor_group:
            permuted = tuple(b[perm[j]] for j in range(h11))
            orbit_val.add(permuted)

            # M=0 check: every permuted bundle must be in the solution multiset
            if permuted not in bundle_counts:
                return False

            # Topological invariance verification
            if chi_cache[permuted] != chi_cache[b]:
                return False

        # Ensure multiset invariance: all bundles in orbit must have same multiplicity
        count = bundle_counts[b]
        for ob in orbit_val:
            if bundle_counts[ob] != count:
                return False
            processed_bundles.add(ob)

        # 2. Per-Orbit Euler Characteristic Check
        # The aggregate chi for this block (multiplicity * sum of chi in orbit)
        # must be divisible by the group order.
        orbit_chi = sum(chi_cache[ob] for ob in orbit_val) * count
        if (orbit_chi % gamma) != 0:
            return False

    return True


def process_manifold(cy, h11, g, db, n_workers, raw_path=None):
    print(f"\nProcessing CY #{cy}, h11={h11}, g={g}...")

    if raw_path is None:
        raw_path = f"Sol_Runs/raw_cy_{cy}_h11_{h11}_g_{g}.jsonl"

    if not os.path.exists(raw_path):
        print(f"  Error: {raw_path} not found.")
        return None

    # Look up CY entry in full database
    cy_entry = next((e for e in db if e["Num"] == cy), None)
    if cy_entry is None:
        print(f"  Error: CY Num {cy} not found in database.")
        return None

    # Determine equivariance mode
    mode, divisor_group = get_divisor_action_info(cy_entry, g)
    print(f"  Equivariance mode: {mode}")
    if mode == "nontrivial":
        print(f"  Divisor group order: {len(divisor_group)}")

    # Count lines to split work
    line_count = 0
    with open(raw_path, "rb") as f:
        for _ in f:
            line_count += 1

    if line_count == 0:
        print("  No matrices found in raw file.")
        return None

    gSym, _ = build_symmetry_group(cy, db)

    # 1. Parallel Scan and Canonicalisation
    print(f"  Scanning and canonicalising {line_count} matrices...")
    batch_size = max(1, line_count // n_workers)
    scan_args = []
    for i in range(0, line_count, batch_size):
        scan_args.append((raw_path, i, min(i + batch_size, line_count), gSym))

    all_events = []  # (h_c, h_g, h_f)
    global_representative_matrices = {}  # h_c -> matrix

    with Pool(n_workers) as p:
        for events, reps in p.imap_unordered(scan_and_canonicalise_worker, scan_args):
            all_events.extend(events)
            for h_c, m in reps.items():
                if h_c not in global_representative_matrices:
                    global_representative_matrices[h_c] = m

    gc.collect()

    # 2. Extract Stats
    n_total = len(all_events)
    unique_h_c = sorted(list(set(e[0] for e in all_events)))
    unique_h_g = set(e[1] for e in all_events)
    unique_h_f = set(e[2] for e in all_events)

    n_unique_col = len(unique_h_c)
    n_unique_geom = len(unique_h_g)
    n_unique_full = len(unique_h_f)

    # 3. Parallel Equivariance on UNIQUE physical configurations (h_c)
    matrices_to_check = [global_representative_matrices[h] for h in unique_h_c]

    print(
        f"  Running equivariance checks on {len(matrices_to_check)} unique physical configurations..."
    )

    # Build CICY configuration for pyCICY
    cicy_conf = cy_entry["Conf"]
    cicy_conf_with_dims = [[sum(row) - 1] + [int(x) for x in row] for row in cicy_conf]

    unique_equiv_results = []
    with Pool(
        n_workers,
        initializer=init_worker,
        initargs=(cicy_conf_with_dims, mode, divisor_group),
    ) as p:
        tasks = [(m, g) for m in matrices_to_check]
        for res in tqdm(
            p.imap(check_equivariance, tasks),
            total=len(tasks),
            desc=f"  Equivariance (CY {cy}, g={g})",
            leave=False,
        ):
            unique_equiv_results.append(res)

    # Map results back to unique h_c
    h_c_to_equiv = {h: unique_equiv_results[i] for i, h in enumerate(unique_h_c)}

    # 4. Collect Final Stats
    n_total_equiv = sum(1 for e in all_events if h_c_to_equiv[e[0]])

    h_g_to_equiv = {}
    h_f_to_equiv = {}
    for e in all_events:
        h_c, h_g, h_f = e
        equiv = h_c_to_equiv[h_c]
        if h_g not in h_g_to_equiv:
            h_g_to_equiv[h_g] = equiv
        if h_f not in h_f_to_equiv:
            h_f_to_equiv[h_f] = equiv

    n_unique_col_equiv = sum(1 for h in unique_h_c if h_c_to_equiv[h])
    n_unique_geom_equiv = sum(1 for h in unique_h_g if h_g_to_equiv[h])
    n_unique_full_equiv = sum(1 for h in unique_h_f if h_f_to_equiv[h])

    # Cleanup
    del all_events
    del global_representative_matrices
    gc.collect()

    res_row = {
        "cy": cy,
        "h11": h11,
        "g": g,
        "mode": mode,
        "n_total": n_total,
        "n_unique_col": n_unique_col,
        "n_unique_geom": n_unique_geom,
        "n_unique_full": n_unique_full,
        "n_total_equiv": n_total_equiv,
        "n_unique_col_equiv": n_unique_col_equiv,
        "n_unique_geom_equiv": n_unique_geom_equiv,
        "n_unique_full_equiv": n_unique_full_equiv,
    }
    return res_row


def main():
    parser = argparse.ArgumentParser(
        description="Check equivariance of line bundle solutions under freely acting symmetries."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of worker processes to use (default: {cpu_count()})",
    )
    parser.add_argument(
        "--cy_index",
        type=int,
        nargs="+",
        help="List of CY IDs to process (default: all found in input_dir)",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="databases/full_cicy_database.json",
        help="Path to the CICY database JSON file (default: databases/full_cicy_database.json)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Sol_Runs",
        help="Directory containing raw_cy_*.jsonl files (default: Sol_Runs)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Analysis_Plots/equivariance_stats.csv",
        help="Output path for CSV stats (default: Analysis_Plots/equivariance_stats.csv)",
    )
    args = parser.parse_args()

    with open(args.db_path, "r") as f:
        db = json.load(f)

    out_csv = args.output_csv
    in_dir = args.input_dir

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    raw_files = glob.glob(
        os.path.join(in_dir, "**", "raw_cy_*_h11_*_g_*.jsonl"), recursive=True
    )

    results = []

    # Filter by CY if requested
    if args.cy_index:
        cy_set = set(args.cy_index)
        raw_files = [
            f for f in raw_files if int(re.search(r"cy_(\d+)", f).group(1)) in cy_set
        ]

    raw_files.sort()

    for f in raw_files:
        m = re.search(r"cy_(\d+)_h11_(\d+)_g_(\d+)", f)
        if m:
            cy_id = int(m.group(1))
            h11 = int(m.group(2))
            g = int(m.group(3))

            row = process_manifold(cy_id, h11, g, db, args.workers, raw_path=f)
            if row:
                # Capture ttype from the directory structure if it is nested
                path_parts = f.split(os.sep)
                in_dir_parts = in_dir.rstrip(os.sep).split(os.sep)
                if len(path_parts) > len(in_dir_parts) + 1:
                    row["tl_type"] = path_parts[len(in_dir_parts)]
                results.append(row)
                # Intermediate save
                pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"\nDone! Final results saved to {out_csv}")


if __name__ == "__main__":
    main()
