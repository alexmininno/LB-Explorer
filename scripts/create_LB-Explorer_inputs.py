#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import re
import sympy as sp

def parse_polynomial_to_tensor(poly_str, h11):
    if h11 == 0:
        return np.zeros((0, 0, 0))

    s = re.sub(r"J\((\d+)\)", r"J_\1", poly_str)
    s = s.replace("^", "**")

    if not s.strip() or s.strip() == "0":
        return np.zeros((h11, h11, h11))

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
                    kappa[i, j, k] = val
                    kappa[i, k, j] = val
                    kappa[j, i, k] = val
                    kappa[j, k, i] = val
                    kappa[k, i, j] = val
                    kappa[k, j, i] = val
    return kappa

def main():
    parser = argparse.ArgumentParser(description="Generate LB-Explorer inputs from full CICY database")
    parser.add_argument("--db_path", type=str, default="databases/full_cicy_database.json", help="Path to input full_cicy_database.json (default: databases/full_cicy_database.json)")
    parser.add_argument("--output_dir", type=str, default="cy_geometry_exports", help="Directory to save generated geometry inputs (default: cy_geometry_exports)")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        print(f"[!] Input database not found: {args.db_path}")
        return

    with open(args.db_path, "r") as f:
        db = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    manifolds_by_h11 = {}

    print(f"[*] Processing {len(db)} entries from {args.db_path}...")
    for entry in db:
        cy_id = entry.get("Num", entry.get("id"))
        h11 = entry.get("H11", entry.get("h11"))
        if cy_id is None or h11 is None:
            continue

        c2_tx = entry.get("C2", entry.get("c2_tx", []))
        poly_str = entry.get("Ring", entry.get("intersecting_polynomial", "0"))
        poly_str = str(poly_str)
        kappa = parse_polynomial_to_tensor(poly_str, h11)
        
        intersection_numbers = []
        for i in range(h11):
            for j in range(i, h11):
                for k in range(j, h11):
                    val = kappa[i, j, k]
                    if val != 0:
                        intersection_numbers.append([i+1, j+1, k+1, int(val)])

        gammas = entry.get("gamma", [])
        if not gammas:
            fas = entry.get("Mapped Freely Acting Symmetries", [])
            gammas = sorted(list(set([s[1] for s in fas]))) if fas else [1]

        geo_data = {
            "index_in_database": cy_id,
            "gamma": gammas,
            "h11": h11,
            "intersection_numbers": intersection_numbers,
            "c2_tx": c2_tx
        }

        if h11 not in manifolds_by_h11:
            manifolds_by_h11[h11] = []
        manifolds_by_h11[h11].append(geo_data)

    for h11, data_list in manifolds_by_h11.items():
        out_path = os.path.join(args.output_dir, f"all_geometry_h11_{h11}.json")
        with open(out_path, "w") as f:
            json.dump(data_list, f, indent=4)
    
    print(f"[*] Successfully generated inputs in {args.output_dir}")

if __name__ == "__main__":
    main()
