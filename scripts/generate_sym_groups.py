import ast
import re
import sympy as sp
from collections import defaultdict
try:
    from sage.all import Graph, SymmetricGroup, gap, Matrix
except ImportError:
    # Handle gracefully so --help can still be accessed
    Graph = SymmetricGroup = gap = Matrix = None

def get_topological_symmetry(c2_vector, kappa_tensor):
    n = len(c2_vector)
    if n <= 1:
        return 1, "1", []

    G = Graph(multiedges=False)
    divisor_nodes = list(range(1, n + 1))
    G.add_vertices(divisor_nodes)

    c2_groups = defaultdict(list)
    for i, val in enumerate(c2_vector):
        c2_groups[val].append(i + 1)

    partitions = list(c2_groups.values())
    kappa_groups = defaultdict(list)

    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                val = kappa_tensor[i][j][k]
                if val != 0:
                    node_name = f"tensor_{i+1}_{j+1}_{k+1}"
                    G.add_vertex(node_name)
                    kappa_groups[val].append(node_name)

                    mults = defaultdict(int)
                    mults[i + 1] += 1
                    mults[j + 1] += 1
                    mults[k + 1] += 1

                    for div_node, multiplicity in mults.items():
                        G.add_edge(node_name, div_node, label=multiplicity)

    partitions.extend(list(kappa_groups.values()))

    full_group = G.automorphism_group(partition=partitions, edge_labels=True)

    S_row = SymmetricGroup(n)
    row_gens = []

    for g in full_group.gens():
        mapping = [g(v) for v in divisor_nodes]
        row_gens.append(S_row(mapping))

    if not row_gens:
        top_group = S_row.subgroup([S_row.one()])
    else:
        top_group = S_row.subgroup(row_gens)

    order = top_group.order()
    structure = "1" if order == 1 else str(gap(top_group).StructureDescription())

    # --- ADDED: Extract the generators as disjoint cycle tuples ---
    generators = top_group.gens()
    explicit_gens = [gen.cycle_tuples() for gen in generators]

    # Clean up empty tuples for the trivial group (identity)
    if not explicit_gens or (len(explicit_gens) == 1 and not explicit_gens[0]):
        explicit_gens = []

    return int(order), structure, explicit_gens


def get_configuration_symmetry(matrix_data):
    M = Matrix(matrix_data)
    num_rows = M.nrows()
    num_cols = M.ncols()
    
    if num_rows == 0 or num_cols == 0:
        return 1, "1", []
        
    row_nodes = list(range(1, num_rows + 1))
    col_nodes = list(range(num_rows + 1, num_rows + num_cols + 1))
    
    G = Graph(multiedges=False)
    G.add_vertices(row_nodes)
    G.add_vertices(col_nodes)
    
    for i in range(num_rows):
        for j in range(num_cols):
            if M[i,j] > 0:
                G.add_edge(row_nodes[i], col_nodes[j], label=int(M[i,j]))
                
    full_group = G.automorphism_group(edge_labels=True, partition=[row_nodes, col_nodes])
    
    S_row = SymmetricGroup(num_rows)
    row_gens = []
    
    for g in full_group.gens():
        restricted_mapping = [g(i) for i in row_nodes]
        row_gens.append(S_row(restricted_mapping))
        
    if not row_gens:
        row_group = S_row.subgroup([S_row.one()])
    else:
        row_group = S_row.subgroup(row_gens)
        
    order = row_group.order()
    structure = "1" if order == 1 else str(gap(row_group).StructureDescription())
    
    generators = row_group.gens()
    explicit_gens = [gen.cycle_tuples() for gen in generators]
    if not explicit_gens or (len(explicit_gens) == 1 and not explicit_gens[0]):
        explicit_gens = []
        
    return int(order), structure, explicit_gens


def parse_polynomial_to_tensor(poly_str, h11):
    if h11 == 0:
        return []

    s = re.sub(r"J\((\d+)\)", r"J_\1", poly_str)
    s = s.replace("^", "**")

    if not s.strip() or s.strip() == "0":
        return [[[0 for _ in range(h11)] for _ in range(h11)] for _ in range(h11)]

    J_vars = sp.symbols(f"J_1:{h11+1}")
    if h11 == 1:
        J_vars = (J_vars,)

    expr = sp.expand(sp.sympify(s))
    kappa = [[[0 for _ in range(h11)] for _ in range(h11)] for _ in range(h11)]

    for i in range(h11):
        for j in range(i, h11):
            for k in range(j, h11):
                term = J_vars[i] * J_vars[j] * J_vars[k]
                c = expr.coeff(term)

                if c != 0:
                    val = int(c)
                    kappa[i][j][k] = val
                    kappa[i][k][j] = val
                    kappa[j][i][k] = val
                    kappa[j][k][i] = val
                    kappa[k][i][j] = val
                    kappa[k][j][i] = val

    return kappa


def process_row(row):
    try:
        if len(row) < 6:
            return None

        cy_id = row[0]
        h11 = int(row[1])
        h21 = int(row[2])
        c2_vector = ast.literal_eval(row[3])
        poly_str = row[4]

        try:
            discrete_sym = ast.literal_eval(row[5])
        except (ValueError, SyntaxError):
            discrete_sym = row[5]

        kappa_tensor = parse_polynomial_to_tensor(poly_str, h11)

        # --- ADDED: Unpack generators and include in dictionary ---
        order, structure, generators = get_topological_symmetry(c2_vector, kappa_tensor)

        # Build configuration matrix symmetries
        order_conf = 1
        structure_conf = "1"
        generators_conf = []
        discrete_sym_conf = [] # Not directly available in the 7-column CSV
        
        if len(row) > 6:
            try:
                conf_matrix = ast.literal_eval(row[6])
                order_conf, structure_conf, generators_conf = get_configuration_symmetry(conf_matrix)
            except Exception:
                pass

        return {
            "CY": cy_id,
            "h11": h11,
            "h21": h21,
            "c2": c2_vector,
            "intersecting_polynomial": poly_str,
            "discrete_symmetries": discrete_sym,
            "group_structure": structure,
            "group_order": order,
            "group_generators": generators,
            "discrete_symmetries_conf": discrete_sym_conf,
            "group_structure_conf": structure_conf,
            "group_order_conf": order_conf,
            "group_generators_conf": generators_conf,
        }
    except Exception as e:
        return {"CY": row[0] if len(row) > 0 else "Unknown", "error": str(e)}

import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Generate topological and configuration symmetry groups for CICYs. NOTE: Running this script requires SageMath to be installed on your system.")
    parser.add_argument("--db_path", type=str, default="databases/full_cicy_database.json", help="Path to input CICY database JSON (default: databases/full_cicy_database.json)")
    parser.add_argument("--output_path", type=str, default="databases/cicy_symmetries.json", help="Path to save output JSON with symmetries (default: databases/cicy_symmetries.json)")
    args = parser.parse_args()

    if Graph is None:
        print("CRITICAL ERROR: SageMath is not installed or accessible in this environment.")
        print("Please run this script using a SageMath python environment.")
        return

    if not os.path.exists(args.db_path):
        print(f"[!] Database {args.db_path} not found.")
        return

    with open(args.db_path, "r") as f:
        db = json.load(f)

    results = []
    print(f"[*] Processing {len(db)} entries from {args.db_path}...")
    for entry in db:
        cy_id = entry.get("Num", entry.get("id", "Unknown"))
        h11 = entry.get("H11", entry.get("h11", 0))
        h21 = entry.get("H21", entry.get("h21", 0))
        c2_vector = entry.get("C2", entry.get("c2_tx", []))
        poly_str = entry.get("Ring", entry.get("intersecting_polynomial", "0"))
        discrete_sym = entry.get("discrete_symmetries", [])
        conf_matrix = entry.get("Conf", [])
        
        row = [cy_id, h11, h21, str(c2_vector), poly_str, str(discrete_sym), str(conf_matrix)]
        res = process_row(row)
        if res:
            results.append(res)
            
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"[*] Processed {len(results)} entries. Saved to {args.output_path}")

if __name__ == "__main__":
    main()
