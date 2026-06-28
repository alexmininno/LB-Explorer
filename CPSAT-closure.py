"""
CLI over cpsat_closure.close_with_cpsat.

Usage:
    python CPSAT-closure.py \\
        --solutions solutions_gpu_h11_5_idx_7447_..._sumoff.jsonl \\
        --geometry cy_geometry_exports/all_geometry_h11_5.json \\
        --cy_index 7447 --gamma 2 \\
        --time_limit 5 \\
        --max_matrices 1000 \\
        --stability_mode lazy \\
        --workers 8 \\
        --output_dir some_dir

Reads K-matrices from the solutions JSON, calls close_with_cpsat on each,
writes results to closed_<input_stem>.json.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from ortools.sat.python import cp_model


def _canonical(K):
    """Column-sorted canonical hash (columns are a direct sum, commutative)."""
    return tuple(sorted(map(tuple, np.asarray(K, dtype=int).T.tolist())))


def _lex_argsort_cols(K):
    """Return column indices that sort K's columns ascending lexicographically."""
    K = np.asarray(K, dtype=int)
    cols = [tuple(K[:, a].tolist()) for a in range(K.shape[1])]
    return sorted(range(len(cols)), key=lambda a: cols[a])


def _add_column_lex(model, K_vars, rank, h11):
    """Enforce K[:, a] <=_lex K[:, a+1] for every adjacent column pair.

    Breaks the rank! column-permutation symmetry of the K-matrix (columns are
    interchangeable because a direct sum of line bundles is commutative).
    """
    for a in range(rank - 1):
        col_a = [K_vars[i][a] for i in range(h11)]
        col_b = [K_vars[i][a + 1] for i in range(h11)]
        eq = [model.NewBoolVar(f"lex_eq_{a}_{i}") for i in range(h11)]
        lt = [model.NewBoolVar(f"lex_lt_{a}_{i}") for i in range(h11)]
        for i in range(h11):
            model.Add(col_a[i] == col_b[i]).OnlyEnforceIf(eq[i])
            model.Add(col_a[i] != col_b[i]).OnlyEnforceIf(eq[i].Not())
            model.Add(col_a[i] < col_b[i]).OnlyEnforceIf(lt[i])
            model.Add(col_a[i] >= col_b[i]).OnlyEnforceIf(lt[i].Not())
        # For each row i: if every prior row is equal, then row i must be
        # equal or strictly less. Clause form:
        #   lt[i]  OR  eq[i]  OR  (some j < i has NOT eq[j]).
        for i in range(h11):
            clause = [lt[i], eq[i]] + [eq[j].Not() for j in range(i)]
            model.AddBoolOr(clause)


class _IncumbentCollector(cp_model.CpSolverSolutionCallback):
    """Records every improving K' CP-SAT finds during Solve().

    CP-SAT emits the callback once per incumbent; calling it across multiple
    Solve() invocations (as the lazy-Stab loop does) accumulates across all.
    Each entry is (K as np.ndarray, objective value as int).
    """

    def __init__(self, K_vars, h11, rank):
        super().__init__()
        self._K_vars = K_vars
        self._h11 = h11
        self._rank = rank
        self.solutions = []

    def OnSolutionCallback(self):
        K = np.zeros((self._h11, self._rank), dtype=int)
        for i in range(self._h11):
            for a in range(self._rank):
                K[i, a] = self.Value(self._K_vars[i][a])
        try:
            obj = int(self.ObjectiveValue())
        except Exception:
            obj = None
        self.solutions.append((K, obj, self.WallTime()))


def _sparse_kappa_entries(kappa):
    """Yield (i, j, k, val) for every nonzero entry in kappa."""
    kappa = np.asarray(kappa)
    nz = np.argwhere(kappa != 0)
    for idx in nz:
        i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
        val = int(round(float(kappa[i, j, k])))
        if val != 0:
            yield i, j, k, val


class _ProductCache:
    """Shared K2 / K3 product-var cache so chi and Anom reuse the same vars.

    Keys:
      K2: (min(i,j), max(i,j), a) -> IntVar equal to K[i,a]·K[j,a]
      K3: (tuple(sorted(i,j,k)), a) -> IntVar equal to K[i,a]·K[j,a]·K[k,a]

    `col_bounds[a]` is the |entry| bound on K[i,a] for column a; product var
    domains are derived from these per-column bounds. When uniform (all
    col_bounds[a] == m_bound), this matches the original symmetric behavior.
    """

    def __init__(self, model, K_vars, m_bound, col_bounds=None):
        self.model = model
        self.K_vars = K_vars
        self.m_bound = m_bound  # legacy; retained for backward compat
        rank = len(K_vars[0]) if K_vars else 0
        self.col_bounds = (
            list(col_bounds) if col_bounds is not None else [m_bound] * rank
        )
        self.K2 = {}
        self.K3 = {}

    def get_K2(self, i, j, a):
        key = (min(i, j), max(i, j), a)
        if key not in self.K2:
            b = self.col_bounds[a] ** 2
            v = self.model.NewIntVar(-b, b, f"K2_{key[0]}_{key[1]}_{a}")
            self.model.AddMultiplicationEquality(
                v, [self.K_vars[i][a], self.K_vars[j][a]]
            )
            self.K2[key] = v
        return self.K2[key]

    def get_K3(self, i, j, k, a):
        key = (tuple(sorted((i, j, k))), a)
        if key not in self.K3:
            ii, jj, kk = key[0]
            b = self.col_bounds[a] ** 3
            v = self.model.NewIntVar(-b, b, f"K3_{ii}_{jj}_{kk}_{a}")
            self.model.AddMultiplicationEquality(
                v, [self.get_K2(ii, jj, a), self.K_vars[kk][a]]
            )
            self.K3[key] = v
        return self.K3[key]


def _add_twelve_chi(model, cache, kappa_entries, c2_tx, gamma, rank):
    """
    Create one IntVar `twelve_chi[a]` per column a equal to 12·chi(L_a):
        12·chi(L_a) = 2·Σ κ_{ijk} K[i,a] K[j,a] K[k,a]  +  Σ c2_tx[i] K[i,a].
    All κ / c2_tx are integers, so twelve_chi is integer-valued.

    Returns list[IntVar] of length rank. Each is bounded to Rng: [-36γ, 0].
    """
    h11 = len(cache.K_vars)
    c2_tx = np.asarray(c2_tx).astype(int).tolist()

    g = int(abs(gamma))
    lo, hi = -36 * g, 0
    safe_bound = max(abs(lo), abs(hi)) + 1

    twelve_chi = []
    for a in range(rank):
        cubic_terms = [
            2 * kval * cache.get_K3(i, j, k, a) for (i, j, k, kval) in kappa_entries
        ]
        linear_terms = [c2_tx[i] * cache.K_vars[i][a] for i in range(h11)]

        tc = model.NewIntVar(-safe_bound, safe_bound, f"twelve_chi_{a}")
        model.Add(tc == sum(cubic_terms) + sum(linear_terms))
        model.Add(tc >= lo)
        model.Add(tc <= hi)
        twelve_chi.append(tc)

    return twelve_chi


def _add_anom(model, cache, kappa_entries, c2_tx, rank):
    """
    Anomaly cancellation: c2(V)_i <= c2(TX)_i for every i, where
        c2(V)_i = 0.5 · Σ_{j,k} κ_{ijk} · (K K^T)[j,k]
                = 0.5 · Σ_{j,k,a} κ_{ijk} · K[j,a] · K[k,a]
    Scale by 2 to stay integer:
        2·c2V[i] = Σ_{j,k} κ_{ijk} · KK[j,k]
    where KK[j,k] = Σ_a K[j,a]·K[k,a] = Σ_a K2[j,k,a] (reused from the chi cache).
    Constraint: 2·c2V[i] <= 2·c2_tx[i].

    Returns list[IntVar] of length h11 holding 2·c2V (handy for diagnostics).
    """
    h11 = len(cache.K_vars)
    c2_tx = np.asarray(c2_tx).astype(int).tolist()

    # Bound: 2·c2V[i] = Σ_{j,k} κ · Σ_a K2. |K2[a]| ≤ col_bounds[a]^2; use the
    # widest col_bound for the safe bound so it covers every column.
    max_kappa = max((abs(v) for *_, v in kappa_entries), default=1)
    max_b = max(cache.col_bounds) if cache.col_bounds else cache.m_bound
    safe = abs(max_kappa) * (h11**2) * rank * (max_b**2) + 1

    # Collect κ[i,j,k] entries grouped by i.
    by_i = [[] for _ in range(h11)]
    for i, j, k, kval in kappa_entries:
        by_i[i].append((j, k, kval))

    two_c2V = []
    for i in range(h11):
        # 2·c2V[i] = Σ_{(j,k) nz under i} κ_{ijk} · Σ_a K2[j,k,a]
        terms = []
        for j, k, kval in by_i[i]:
            for a in range(rank):
                terms.append(kval * cache.get_K2(j, k, a))
        v = model.NewIntVar(-safe, safe, f"two_c2V_{i}")
        model.Add(v == sum(terms))
        model.Add(v <= 2 * c2_tx[i])
        two_c2V.append(v)
    return two_c2V


class _SProductCache:
    """Product cache for pair-column vars S[i, p] = K[i, a_p] + K[i, b_p].

    `pair_bounds[p]` is the |entry| bound on S[i, p]: pair_bounds[p] =
    col_bounds[a] + col_bounds[b] for the (a, b) pair indexed by p. Product
    var domains derive from per-pair bounds. When all pair_bounds are equal to
    `2 * m_bound`, this matches the original symmetric behavior.
    """

    def __init__(self, model, S_vars, m_bound, pair_bounds=None):
        self.model = model
        self.S_vars = S_vars
        self.m_bound = m_bound  # legacy; retained for backward compat
        n_pairs = len(S_vars[0]) if (S_vars and S_vars[0]) else 0
        self.pair_bounds = (
            list(pair_bounds) if pair_bounds is not None else [2 * m_bound] * n_pairs
        )
        self.S2 = {}
        self.S3 = {}

    def get_S2(self, i, j, p):
        key = (min(i, j), max(i, j), p)
        if key not in self.S2:
            b = self.pair_bounds[p] ** 2
            v = self.model.NewIntVar(-b, b, f"S2_{key[0]}_{key[1]}_{p}")
            self.model.AddMultiplicationEquality(
                v, [self.S_vars[i][p], self.S_vars[j][p]]
            )
            self.S2[key] = v
        return self.S2[key]

    def get_S3(self, i, j, k, p):
        key = (tuple(sorted((i, j, k))), p)
        if key not in self.S3:
            ii, jj, kk = key[0]
            b = self.pair_bounds[p] ** 3
            v = self.model.NewIntVar(-b, b, f"S3_{ii}_{jj}_{kk}_{p}")
            self.model.AddMultiplicationEquality(
                v, [self.get_S2(ii, jj, p), self.S_vars[kk][p]]
            )
            self.S3[key] = v
        return self.S3[key]


def _add_pair(
    model, K_vars, kappa_entries, c2_tx, gamma, rank, m_bound, col_bounds=None
):
    """
    For every pair of columns (a, b) with a<b:
      1. Create S[i, p] = K[i, a] + K[i, b] for i=0..h11-1.
      2. Build 12·chi_pair[p] = 2·Σ κ_{ijk} S[i,p] S[j,p] S[k,p]
                               + Σ c2_tx[i] S[i,p].
      3. Enforce -36γ ≤ 12·chi_pair[p] ≤ 0 (Pair + Rng on column sums).

    `col_bounds` (len rank): per-column |entry| bound on K[i, a]. Used to
    derive `pair_bounds[p] = col_bounds[a] + col_bounds[b]`. If None, defaults
    to uniform [m_bound]*rank → all pair_bounds = 2*m_bound (legacy).

    Returns (S_vars, twelve_chi_pair, pairs_list), where pairs_list is the
    ordered list [(a, b), ...] used for indexing p.
    """
    h11 = len(K_vars)
    c2_tx = np.asarray(c2_tx).astype(int).tolist()
    pairs = [(a, b) for a in range(rank) for b in range(a + 1, rank)]
    n_pairs = len(pairs)
    if col_bounds is None:
        col_bounds = [m_bound] * rank
    pair_bounds = [col_bounds[a] + col_bounds[b] for (a, b) in pairs]

    # S[i, p] int vars (per-pair bound to accommodate asymmetric col_bounds).
    S_vars = [
        [
            model.NewIntVar(-pair_bounds[p], pair_bounds[p], f"S_{i}_{p}")
            for p in range(n_pairs)
        ]
        for i in range(h11)
    ]
    for i in range(h11):
        for p, (a, b) in enumerate(pairs):
            model.Add(S_vars[i][p] == K_vars[i][a] + K_vars[i][b])

    scache = _SProductCache(model, S_vars, m_bound, pair_bounds=pair_bounds)
    g = int(abs(gamma))
    lo, hi = -36 * g, 0
    safe_bound = max(abs(lo), abs(hi)) + 1

    twelve_chi_pair = []
    for p in range(n_pairs):
        cubic_terms = [
            2 * kval * scache.get_S3(i, j, k, p) for (i, j, k, kval) in kappa_entries
        ]
        linear_terms = [c2_tx[i] * S_vars[i][p] for i in range(h11)]
        tc = model.NewIntVar(-safe_bound, safe_bound, f"twelve_chi_pair_{p}")
        model.Add(tc == sum(cubic_terms) + sum(linear_terms))
        model.Add(tc >= lo)
        model.Add(tc <= hi)
        twelve_chi_pair.append(tc)

    return S_vars, twelve_chi_pair, pairs


def _add_ntr(model, K_vars, S_vars, rank, m_bound, col_bounds=None, pair_bounds=None):
    """
    Non-triviality:
      - No column all-zero:    Σ_i |K[i, a]| ≥ 1  for every a.
      - No pair cancellation:  Σ_i |S[i, p]| ≥ 1  for every pair p.

    Creates abs-value aux vars for every K and S entry (distinct from
    abs_delta, which is |K - K0|). Bounds derive from `col_bounds[a]` /
    `pair_bounds[p]` to support asymmetric K[i,a] domains.
    """
    h11 = len(K_vars)
    n_pairs = len(S_vars[0]) if S_vars else 0
    if col_bounds is None:
        col_bounds = [m_bound] * rank
    if pair_bounds is None and S_vars:
        pair_bounds = [2 * m_bound] * n_pairs

    abs_K = [
        [model.NewIntVar(0, col_bounds[a], f"absK_{i}_{a}") for a in range(rank)]
        for i in range(h11)
    ]
    for i in range(h11):
        for a in range(rank):
            model.AddAbsEquality(abs_K[i][a], K_vars[i][a])
    for a in range(rank):
        model.Add(sum(abs_K[i][a] for i in range(h11)) >= 1)

    if S_vars:
        abs_S = [
            [
                model.NewIntVar(0, pair_bounds[p], f"absS_{i}_{p}")
                for p in range(n_pairs)
            ]
            for i in range(h11)
        ]
        for i in range(h11):
            for p in range(n_pairs):
                model.AddAbsEquality(abs_S[i][p], S_vars[i][p])
        for p in range(n_pairs):
            model.Add(sum(abs_S[i][p] for i in range(h11)) >= 1)


def _probe_vectors(rank, stability_range=2):
    """Replicates the gpu_only.py probe-vector set:
    all v in [-R, R]^rank except v=0 and v proportional to (1,1,...,1).
    For rank=5, stability_range=2 this yields 3120 vectors.
    """
    from itertools import product

    vecs = []
    for v in product(range(-stability_range, stability_range + 1), repeat=rank):
        if all(x == 0 for x in v):
            continue
        if all(x == v[0] for x in v):
            continue
        vecs.append(v)
    return vecs


def _add_one_probe_stab(
    model,
    K_vars,
    kappa_entries,
    v,
    rank,
    h11,
    m_bound,
    stability_range=2,
    col_bounds=None,
):
    """Add indefiniteness constraints for a single probe vector v.

    Returns the number of P_v auxiliary vars created (0 if the vector has
    empty support under κ). `col_bounds` widens the safe bound to account for
    asymmetric K[i,a] domains; falls back to uniform `m_bound` if None.
    """
    by_jk = {}
    for i, j, k, kval in kappa_entries:
        by_jk.setdefault((j, k), []).append((i, kval))
    max_kappa = max((abs(x) for *_, x in kappa_entries), default=1)
    max_b = max(col_bounds) if col_bounds is not None else m_bound
    safe = max_kappa * h11 * rank * stability_range * max_b + 1

    def Kv_expr(i):
        return sum(v[a] * K_vars[i][a] for a in range(rank))

    P_vars = []
    tag = "_".join(str(x) for x in v).replace("-", "m")
    for (j, k), entries in by_jk.items():
        p_var = model.NewIntVar(-safe, safe, f"Pv{tag}_{j}_{k}")
        model.Add(p_var == sum(kval * Kv_expr(i) for (i, kval) in entries))
        P_vars.append(p_var)
    if not P_vars:
        return 0
    mx = model.NewIntVar(-safe, safe, f"Pvmax_{tag}")
    mn = model.NewIntVar(-safe, safe, f"Pvmin_{tag}")
    model.AddMaxEquality(mx, P_vars)
    model.AddMinEquality(mn, P_vars)
    model.Add(mx >= 1)
    model.Add(mn <= -1)
    return len(P_vars)


def _numpy_stab_violators(K, kappa, probes):
    """Return list of probe vectors v for which P_v fails indefiniteness."""
    K = np.asarray(K, dtype=float)
    kappa_f = np.asarray(kappa, dtype=float)
    bad = []
    for v in probes:
        Kv = np.einsum("a,ia->i", np.array(v, dtype=float), K)
        P = np.einsum("ijk,i->jk", kappa_f, Kv)
        if P.max() < 1 - 1e-6 or P.min() > -1 + 1e-6:
            bad.append(v)
    return bad


def _add_stab_eager(
    model,
    K_vars,
    kappa,
    kappa_entries,
    rank,
    m_bound,
    stability_range=2,
    col_bounds=None,
):
    """
    Encode Bogomolov stability for EVERY probe vector v:
      P_v[j, k] = Σ_i κ_{ijk} · (Σ_a v[a] · K[i, a])    (linear in K!)
    must be indefinite, i.e. ∃(j,k): P_v[j,k] ≥ 1  AND  ∃(j',k'): P_v[j',k'] ≤ -1.

    We enforce this via max(P_v) ≥ 1 and min(P_v) ≤ -1 using AddMaxEquality /
    AddMinEquality on the flattened (j,k) grid.

    For rank=5 / range=2 there are 3120 probe vectors → 3120 · h11^2 linear
    aux vars + 2 · 3120 min/max aux vars. Heavy but tractable up to h11≈6.

    `col_bounds` propagates per-column bound info to the safe aux-var bound;
    falls back to uniform `m_bound` if None.
    """
    h11 = len(K_vars)
    probes = _probe_vectors(rank, stability_range)
    n_added = 0
    for v in probes:
        if (
            _add_one_probe_stab(
                model,
                K_vars,
                kappa_entries,
                v,
                rank,
                h11,
                m_bound,
                stability_range,
                col_bounds=col_bounds,
            )
            > 0
        ):
            n_added += 1
    return n_added


def _build_model(
    K0,
    kappa=None,
    c2_tx=None,
    gamma=None,
    m_bound=8,
    perturbation_budget=None,
    stability_mode="eager",
    apply_column_lex=True,
    tail_col_bound=None,
    objective_cols=None,
):
    """
    Build the CP-SAT model.
      - integer K[i, a] in [-col_bounds[a], +col_bounds[a]]
        where col_bounds = [m_bound]*(rank-1) + [tail_col_bound]
        (default tail_col_bound = m_bound → uniform symmetric box)
      - RowSum: sum over cols of K[i, :] == 0 for every row i
      - Sum + Rng via 12·chi linearization (requires kappa, c2_tx, gamma)
      - Optional L1 budget: sum |K[i,a] - K0[i,a]| over `objective_cols`
        <= perturbation_budget
      - Objective: minimize sum of |K[i,a] - K0[i,a]| over `objective_cols`
        (default = all cols).

    `tail_col_bound` widens column rank-1's domain (mirrors engine semantics
    where the last column is RowSum-derived and may exceed m_bound under
    `ignore_bounds=True`). `objective_cols` (iterable of column indices)
    restricts the L1 sum to a subset; col rank-1 is dropped to leave col 4 a
    "free consequence" of cols 0..rank-2.

    If kappa / c2_tx / gamma are all None, the physics constraints are omitted
    (pure structural mode, useful for unit-testing the skeleton).

    Returns (model, vars_dict). vars_dict holds 'K' (list of lists of IntVar),
    'abs_delta' (same shape), 'col_bounds' (list of len rank), and 'twelve_chi'
    (list of IntVar, rank) when physics is active.
    """
    K0 = np.asarray(K0, dtype=int)
    h11, rank = K0.shape

    # Asymmetric per-column bounds.
    if tail_col_bound is None:
        tail_col_bound = m_bound
    col_bounds = [m_bound] * max(0, rank - 1) + [tail_col_bound]
    if rank == 0:
        col_bounds = []
    # Resolve objective columns (default = every column).
    if objective_cols is None:
        obj_cols = list(range(rank))
    else:
        obj_cols = sorted(set(int(c) for c in objective_cols))
        for c in obj_cols:
            if not (0 <= c < rank):
                raise ValueError(f"objective_cols entry {c} out of range [0, {rank})")

    model = cp_model.CpModel()

    # Decision variables, asymmetric bounds.
    K = [
        [
            model.NewIntVar(-col_bounds[a], col_bounds[a], f"K_{i}_{a}")
            for a in range(rank)
        ]
        for i in range(h11)
    ]

    # RowSum == 0, per row.
    for i in range(h11):
        model.Add(sum(K[i][a] for a in range(rank)) == 0)

    # Column-lex symmetry breaking (rank! permutation symmetry).
    # Note: lex-ordering is incompatible with asymmetric col_bounds (would
    # force the high-magnitude col to be lex-greater regardless of physics).
    # Caller is responsible for not requesting both.
    if apply_column_lex and rank > 1:
        _add_column_lex(model, K, rank, h11)

    # L1-distance auxiliary vars.
    delta = [
        [
            model.NewIntVar(-2 * col_bounds[a], 2 * col_bounds[a], f"d_{i}_{a}")
            for a in range(rank)
        ]
        for i in range(h11)
    ]
    abs_delta = [
        [model.NewIntVar(0, 2 * col_bounds[a], f"ad_{i}_{a}") for a in range(rank)]
        for i in range(h11)
    ]
    for i in range(h11):
        for a in range(rank):
            model.Add(delta[i][a] == K[i][a] - int(K0[i, a]))
            model.AddAbsEquality(abs_delta[i][a], delta[i][a])

    # Optional budget cap (restricted to objective cols for semantic
    # consistency with the L1 sum).
    if perturbation_budget is not None:
        model.Add(
            sum(abs_delta[i][a] for i in range(h11) for a in obj_cols)
            <= int(perturbation_budget)
        )

    # Physics: Sum + Rng + Anom + Pair + Ntr + Stab.
    twelve_chi = None
    two_c2V = None
    twelve_chi_pair = None
    S_vars = None
    n_stab = 0
    if kappa is not None and c2_tx is not None and gamma is not None:
        cache = _ProductCache(model, K, m_bound, col_bounds=col_bounds)
        kappa_entries = list(_sparse_kappa_entries(kappa))
        twelve_chi = _add_twelve_chi(
            model, cache, kappa_entries, c2_tx, gamma, rank=rank
        )
        g = int(abs(gamma))
        # Sum: Σ_a 12·chi(L_a) == -36γ.
        model.Add(sum(twelve_chi) == -36 * g)
        # Anom.
        two_c2V = _add_anom(model, cache, kappa_entries, c2_tx, rank=rank)
        # Pair (also adds S vars + 12·chi_pair).
        S_vars, twelve_chi_pair, _pairs = _add_pair(
            model,
            K,
            kappa_entries,
            c2_tx,
            gamma,
            rank=rank,
            m_bound=m_bound,
            col_bounds=col_bounds,
        )
        # Pair bounds (used downstream by _add_ntr for absS bounds).
        pairs_list = [(a, b) for a in range(rank) for b in range(a + 1, rank)]
        pair_bounds = [col_bounds[a] + col_bounds[b] for (a, b) in pairs_list]
        # Ntr on columns and pair-columns.
        _add_ntr(
            model,
            K,
            S_vars,
            rank=rank,
            m_bound=m_bound,
            col_bounds=col_bounds,
            pair_bounds=pair_bounds,
        )
        # Stab.
        if stability_mode == "eager":
            n_stab = _add_stab_eager(
                model,
                K,
                kappa,
                kappa_entries,
                rank=rank,
                m_bound=m_bound,
                col_bounds=col_bounds,
            )
        elif stability_mode == "skip":
            pass
        else:
            raise ValueError(f"unknown stability_mode: {stability_mode!r}")

    # Objective: L1 distance over the chosen subset of columns.
    model.Minimize(sum(abs_delta[i][a] for i in range(h11) for a in obj_cols))

    # Decision strategy: when restricting the objective to a strict subset of
    # columns (i.e. col rank-1 is "free"), tell CP-SAT to never branch on the
    # excluded vars — they are derived through RowSum=0 propagation. Default
    # behavior (full objective) leaves CP-SAT's auto-heuristic intact.
    excluded = [a for a in range(rank) if a not in obj_cols]
    if excluded and rank > 1:
        agent_vars = [K[i][a] for i in range(h11) for a in obj_cols]
        if agent_vars:
            model.AddDecisionStrategy(
                agent_vars,
                cp_model.CHOOSE_FIRST,
                cp_model.SELECT_MIN_VALUE,
            )

    vars_dict = {
        "K": K,
        "abs_delta": abs_delta,
        "twelve_chi": twelve_chi,
        "two_c2V": two_c2V,
        "twelve_chi_pair": twelve_chi_pair,
        "S": S_vars,
        "n_stab_constraints": n_stab,
        "h11": h11,
        "rank": rank,
        "col_bounds": col_bounds,
        "obj_cols": obj_cols,
    }
    return model, vars_dict


def _extract_solution(solver, vars_dict):
    """Pull the K solution out of the solver into a numpy array."""
    K = vars_dict["K"]
    h11, rank = vars_dict["h11"], vars_dict["rank"]
    out = np.zeros((h11, rank), dtype=int)
    for i in range(h11):
        for a in range(rank):
            out[i, a] = solver.Value(K[i][a])
    return out


def _finalize_result(
    status_name,
    collector,
    K0_sorted,
    kappa,
    c2_tx,
    gamma,
    m_bound,
    stability_range,
    stab_constraints,
    stab_iters,
    wall_time_s,
    last_K,
    last_dist,
    tail_col_bound=None,
):
    """Post-verify all collected incumbents against the 8-constraint set, return
    the public result dict with full K_list + distances + wall_times plus best
    K/distance.

    NO incumbent dedup — every post-verified K' from `collector.solutions` is
    kept (including column-permuted siblings). Downstream analysis applies its
    own canonicalization (typically Γ_CY × col-perm full canonical) over the
    union of K_list across many K0s; doing per-solve dedup here would discard
    audit-trail data without affecting the final phys-distinct count.

    `tail_col_bound` is forwarded to `_verify_all_constraints` so the bnd flag
    is checked against the actual asymmetric K-domain used during solving.
    """
    verified = []
    if kappa is not None and c2_tx is not None and gamma is not None:
        for K, obj, wt in collector.solutions:
            flags = _verify_all_constraints(
                K,
                kappa,
                c2_tx,
                gamma,
                m_bound=m_bound,
                stability_range=stability_range,
                tail_col_bound=tail_col_bound,
            )
            if flags["all"]:
                verified.append((K, obj, wt))
    else:
        # Structural-only mode: nothing to verify; pass all through.
        verified = list(collector.solutions)

    verified.sort(key=lambda kv: (kv[1] if kv[1] is not None else 10**18))

    K_list = [K for K, _, _ in verified]
    distances = [obj for _, obj, _ in verified]
    wall_times = [round(wt, 3) for _, _, wt in verified]

    best_K = K_list[0] if K_list else last_K
    best_dist = distances[0] if distances else last_dist

    return {
        "status": status_name,
        "K": best_K,
        "distance": best_dist,
        "K_list": K_list,
        "distances": distances,
        "wall_times": wall_times,
        "stab_constraints": stab_constraints,
        "stab_iters": stab_iters,
        "wall_time_s": wall_time_s,
    }


def close_with_cpsat(
    K0,
    kappa=None,
    c2_tx=None,
    gamma=None,
    *,
    m_bound=8,
    perturbation_budget=None,
    stability_mode="lazy",
    time_limit_s=5,
    workers=8,
    verbose=False,
    hint_K0=True,
    stability_range=2,
    lazy_per_iter_s=30,
    lazy_batch=20,
    apply_column_lex=False,
    tail_col_bound=None,
    objective_cols=None,
):
    """
    Public entry point. Encodes the full 8-constraint model (when kappa,
    c2_tx, gamma are all provided) and minimizes ‖K' - K0_sorted‖_1 (over
    `objective_cols`), where K0_sorted has its columns pre-sorted
    lexicographically when apply_column_lex=True.

    Returns a dict with 'K_list' (all verified 8/8 K' incumbents deduped by
    column-canonical hash and sorted by distance ascending), 'distances'
    (parallel list), plus backward-compat 'K' / 'distance' = the best entry.

    stability_mode:
      - 'lazy' (default): no Stab up front. Solve -> numpy-check which probe
        vectors the incumbent K' violates -> add up to `lazy_batch` of them to
        the model -> re-solve. Repeat until K' passes Stab or time runs out.
      - 'eager': encode all ~3120 probe vectors up front. Correct but slow.
      - 'skip': omit Stab entirely. Post-hoc verification still filters K_list
        to only 8/8-valid K's when physics is available.

    lazy_per_iter_s: time budget per lazy solve (clamped by remaining total).
    lazy_batch: how many violating probes to add per iteration.

    tail_col_bound: |entry| bound for the LAST column (rank-1). Defaults to
      m_bound (uniform symmetric box). Set wider (e.g. (rank-1)*m_bound) when
      K0s have unbounded last col due to engine `ignore_bounds=True` saving.
    objective_cols: iterable of column indices to include in the L1 objective
      (and any perturbation_budget). Defaults to range(rank). Pass
      range(rank-1) to make the last col a "free consequence" of RowSum.
      When restricted, an explicit AddDecisionStrategy keeps CP-SAT from
      branching on the excluded columns.
    """
    import time

    t0 = time.time()
    K0_arr = np.asarray(K0, dtype=int)
    h11, rank = K0_arr.shape

    # Pre-sort K0's columns lex so the objective / hint / lex-constrained K
    # are all consistent. When apply_column_lex is False, K0 passes through
    # untouched so the objective and reported distance use the original K0.
    if apply_column_lex:
        col_perm = _lex_argsort_cols(K0_arr)
        K0_sorted = K0_arr[:, col_perm]
    else:
        K0_sorted = K0_arr

    # When physics args are missing, there is no Stab to encode — downgrade.
    if kappa is None or c2_tx is None or gamma is None:
        if stability_mode != "skip":
            stability_mode = "skip"

    if stability_mode in ("eager", "skip"):
        model, vars_dict = _build_model(
            K0_sorted,
            kappa=kappa,
            c2_tx=c2_tx,
            gamma=gamma,
            m_bound=m_bound,
            perturbation_budget=perturbation_budget,
            stability_mode=stability_mode,
            apply_column_lex=apply_column_lex,
            tail_col_bound=tail_col_bound,
            objective_cols=objective_cols,
        )
        K_vars = vars_dict["K"]
        if hint_K0:
            for i in range(h11):
                for a in range(rank):
                    model.AddHint(K_vars[i][a], int(K0_sorted[i, a]))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_s)
        solver.parameters.num_search_workers = int(workers)
        if verbose:
            solver.parameters.log_search_progress = True
        collector = _IncumbentCollector(K_vars, h11, rank)
        status = solver.Solve(model, collector)
        wall = time.time() - t0
        status_name = solver.StatusName(status)
        last_K = None
        last_dist = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            last_K = _extract_solution(solver, vars_dict)
            last_dist = int(solver.ObjectiveValue())
        return _finalize_result(
            status_name,
            collector,
            K0_sorted,
            kappa,
            c2_tx,
            gamma,
            m_bound,
            stability_range,
            stab_constraints=vars_dict.get("n_stab_constraints", 0),
            stab_iters=0,
            wall_time_s=wall,
            last_K=last_K,
            last_dist=last_dist,
            tail_col_bound=tail_col_bound,
        )

    # -------- stability_mode == 'lazy' ---------------------------------
    if stability_mode != "lazy":
        raise ValueError(f"unknown stability_mode: {stability_mode!r}")
    if kappa is None or c2_tx is None or gamma is None:
        raise ValueError("'lazy' mode requires kappa, c2_tx, gamma")

    # Build model without Stab.
    model, vars_dict = _build_model(
        K0_sorted,
        kappa=kappa,
        c2_tx=c2_tx,
        gamma=gamma,
        m_bound=m_bound,
        perturbation_budget=perturbation_budget,
        stability_mode="skip",
        apply_column_lex=apply_column_lex,
        tail_col_bound=tail_col_bound,
        objective_cols=objective_cols,
    )
    K_vars = vars_dict["K"]
    col_bounds_local = vars_dict.get("col_bounds")
    kappa_entries = list(_sparse_kappa_entries(kappa))
    probes = _probe_vectors(rank, stability_range)

    if hint_K0:
        for i in range(h11):
            for a in range(rank):
                model.AddHint(K_vars[i][a], int(K0_sorted[i, a]))

    # Single collector accumulates incumbents across all lazy iterations.
    collector = _IncumbentCollector(K_vars, h11, rank)

    added_probes = set()
    stab_iters = 0
    last_K = None
    last_dist = None
    last_status = "UNKNOWN"

    while True:
        remaining = time_limit_s - (time.time() - t0)
        if remaining <= 0:
            break
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(min(remaining, lazy_per_iter_s))
        solver.parameters.num_search_workers = int(workers)
        if verbose:
            solver.parameters.log_search_progress = True
        status = solver.Solve(model, collector)
        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            last_status = status_name
            break
        K = _extract_solution(solver, vars_dict)
        dist = int(solver.ObjectiveValue())
        last_K, last_dist, last_status = K, dist, status_name

        # Numpy-check which probes v are violated by this K.
        violators = _numpy_stab_violators(K, kappa, probes)
        violators = [v for v in violators if v not in added_probes]
        if not violators:
            # K satisfies Stab on all probes — we're done.
            break

        # Add up to lazy_batch constraints, newly hint K for warm-start.
        for v in violators[:lazy_batch]:
            _add_one_probe_stab(
                model,
                K_vars,
                kappa_entries,
                v,
                rank,
                h11,
                m_bound,
                stability_range,
                col_bounds=col_bounds_local,
            )
            added_probes.add(v)
        # Warm-start: hint current K (even though it violates the new Stab,
        # it's a good starting point for the incremental LP).
        model.ClearHints()
        for i in range(h11):
            for a in range(rank):
                model.AddHint(K_vars[i][a], int(K[i, a]))
        stab_iters += 1

    wall = time.time() - t0
    return _finalize_result(
        last_status,
        collector,
        K0_sorted,
        kappa,
        c2_tx,
        gamma,
        m_bound,
        stability_range,
        stab_constraints=len(added_probes),
        stab_iters=stab_iters,
        wall_time_s=wall,
        last_K=last_K,
        last_dist=last_dist,
        tail_col_bound=tail_col_bound,
    )


def _numpy_chi(K, kappa, c2_tx):
    """Reference chi computation (floating-point) for test validation."""
    K = np.asarray(K, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    c2_tx = np.asarray(c2_tx, dtype=float)
    cubic = np.einsum("ijk,ia,ja,ka->a", kappa, K, K, K) / 6.0
    linear = np.einsum("i,ia->a", c2_tx, K) / 12.0
    return cubic + linear


def _numpy_c2V(K, kappa):
    """Reference c2(V) = 0.5 · Σ_{j,k} κ_{ijk} · (K K^T)[j,k], shape (h11,)."""
    K = np.asarray(K, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    KKT = K @ K.T
    return 0.5 * np.einsum("ijk,jk->i", kappa, KKT)


def _verify_all_constraints(
    K, kappa, c2_tx, gamma, m_bound=8, stability_range=2, tail_col_bound=None
):
    """Reference check of all 8 constraints. Returns dict of bool flags.

    Used as ground-truth validation of a CP-SAT output against the same
    algebraic conventions as gpu_only.py.

    `tail_col_bound`, if provided, is the per-entry bound on the LAST column
    only (cols 0..rank-2 still capped by `m_bound`). Falls back to uniform
    `m_bound` for backward compat.
    """
    K = np.asarray(K, dtype=int)
    h11, rank = K.shape
    g = int(abs(gamma))
    target = -3 * g

    if tail_col_bound is None:
        col_bounds = np.full(rank, m_bound, dtype=int)
    else:
        col_bounds = np.array(
            [m_bound] * max(0, rank - 1) + [tail_col_bound], dtype=int
        )

    flags = {}
    flags["rowsum"] = bool(np.all(K.sum(axis=1) == 0))
    flags["bnd"] = bool(np.all(np.abs(K) <= col_bounds[None, :]))

    chi = _numpy_chi(K, kappa, c2_tx)
    flags["sum"] = bool(abs(chi.sum() - target) < 1e-6)
    flags["rng"] = bool(np.all((chi >= target - 1e-6) & (chi <= 1e-6)))

    c2V = _numpy_c2V(K, kappa)
    flags["anom"] = bool(np.all(c2V <= c2_tx + 1e-6))

    # Pair: χ of every column-sum also in [target, 0].
    pair_ok = True
    for a in range(rank):
        for b in range(a + 1, rank):
            S = K[:, a] + K[:, b]
            S_col = S[:, None]
            chi_pair = _numpy_chi(S_col, kappa, c2_tx)[0]
            if not (target - 1e-6 <= chi_pair <= 1e-6):
                pair_ok = False
                break
        if not pair_ok:
            break
    flags["pair"] = pair_ok

    # Ntr: no zero column, no pair cancellation.
    ntr_col = all(np.any(K[:, a] != 0) for a in range(rank))
    ntr_pair = all(
        np.any((K[:, a] + K[:, b]) != 0)
        for a in range(rank)
        for b in range(a + 1, rank)
    )
    flags["ntr"] = bool(ntr_col and ntr_pair)

    # Stab: indefinite for every probe vector.
    stab_ok = True
    kappa_f = np.asarray(kappa, dtype=float)
    for v in _probe_vectors(rank, stability_range):
        Kv = np.einsum("a,ia->i", np.array(v, dtype=float), K.astype(float))
        P = np.einsum("ijk,i->jk", kappa_f, Kv)
        if P.max() < 1 - 1e-6 or P.min() > -1 + 1e-6:
            stab_ok = False
            break
    flags["stab"] = stab_ok

    flags["all"] = all(v for k, v in flags.items() if k != "all")
    return flags



def load_geometry(geometry_path, cy_index):
    with open(geometry_path) as f:
        data = json.load(f)
    match = [c for c in data if c.get('index_in_database') == cy_index]
    if not match:
        raise ValueError(f'CY {cy_index} not found in {geometry_path}')
    geo = match[0]
    h11 = geo['h11']
    kappa = np.zeros((h11, h11, h11), dtype=np.int64)
    for entry in geo['intersection_numbers']:
        if len(entry) == 4:
            i, j, k, val = entry
            kappa[i - 1, j - 1, k - 1] = val
    c2_tx = np.array(geo['c2_tx'], dtype=np.int64)
    return kappa, c2_tx, h11


def _canonical(K):
    """Column-sorted canonical hash for dedup (columns are a direct sum,
    commutative)."""
    return tuple(sorted(map(tuple, np.asarray(K, dtype=int).T.tolist())))


def load_solutions(path):
    """Load K-matrices from .json or .jsonl file.
    .json: expects a single JSON array [[K0], [K1], ...].
    .jsonl: one JSON-encoded K-matrix per line.
    """
    if path.endswith('.jsonl'):
        matrices = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    matrices.append(json.loads(line))
        return matrices
    else:
        with open(path) as f:
            return json.load(f)


def process_file(solutions_path, kappa, c2_tx, gamma,
                 time_limit=60, max_matrices=None,
                 stability_mode='eager', workers=8, m_bound=8,
                 apply_column_lex=False, deadline=None,
                 tail_col_bound=None, objective_cols=None):
    """Load solutions JSON/JSONL, run close_with_cpsat on each unique K, return
    list of result dicts plus a `hit_deadline` bool.

    If `deadline` (absolute time.time() value) is set, the per-K0 loop aborts
    before starting a new K0 once the wall clock has passed it.
    """
    sols = load_solutions(solutions_path)
    if max_matrices is not None:
        sols = sols[:max_matrices]

    # NO intra-chunk dedup. Process every K0 in the input order. Chunking is
    # the sole source of dedup; keeping column-permuted siblings here as
    # separate solves maximizes phys-distinct K' discovery via distinct
    # branching paths. Dedup happens at analysis time.
    results = []
    hit_deadline = False
    for idx, K_list in enumerate(sols):
        if deadline is not None and time.time() >= deadline:
            print(f'  [deadline reached before idx={idx}; stopping this file]')
            hit_deadline = True
            break
        K0 = np.array(K_list, dtype=int)

        # Check if K0 already passes all 8 (some Phase-1 saves do).
        # Use the same asymmetric Bnd as the solver so K0s with |K0[i,4]|>m_bound
        # are not falsely flagged as failing Bnd.
        pre_flags = _verify_all_constraints(
            K0, kappa, c2_tx, gamma,
            m_bound=m_bound, tail_col_bound=tail_col_bound)

        # Clamp per-K0 solver time to the remaining overall budget, if any.
        eff_time_limit = time_limit
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                hit_deadline = True
                break
            eff_time_limit = max(1, min(time_limit, int(remaining)))

        t0 = time.time()
        try:
            res = close_with_cpsat(
                K0, kappa=kappa, c2_tx=c2_tx, gamma=gamma,
                stability_mode=stability_mode, m_bound=m_bound,
                time_limit_s=eff_time_limit, workers=workers,
                apply_column_lex=apply_column_lex,
                tail_col_bound=tail_col_bound,
                objective_cols=objective_cols,
            )
        except Exception as e:
            res = {'status': f'ERROR:{e}', 'K': None, 'distance': None,
                   'K_list': [], 'distances': [],
                   'stab_constraints': 0, 'wall_time_s': time.time() - t0}

        res_K_list = res.get('K_list', [])
        res_distances = res.get('distances', [])

        out = {
            'idx': idx,
            'pre_all_pass': pre_flags['all'],
            'pre_flags': {k: v for k, v in pre_flags.items() if k != 'all'},
            'status': res['status'],
            'distance': res['distance'],
            'n_verified': len(res_K_list),
            'wall_time_s': round(res['wall_time_s'], 3),
        }
        if res['K'] is not None:
            K_out = res['K']
            post_flags = _verify_all_constraints(
                K_out, kappa, c2_tx, gamma,
                m_bound=m_bound, tail_col_bound=tail_col_bound)
            out['post_all_pass'] = post_flags['all']
            out['post_flags'] = {k: v for k, v in post_flags.items() if k != 'all'}
            out['K'] = K_out.tolist()
            out['K_list'] = [K.tolist() for K in res_K_list]
            out['distances'] = list(res_distances)
            out['wall_times'] = res.get('wall_times', [])
        else:
            out['post_all_pass'] = False
            out['post_flags'] = None
            out['K'] = None
            out['K_list'] = []
            out['distances'] = []
            out['wall_times'] = []

        print(
            f'  idx={idx:4d} pre_all={out["pre_all_pass"]!s:5} '
            f'status={out["status"]:>12} dist={out["distance"]} '
            f'post_all={out["post_all_pass"]!s:5} '
            f'n_verified={out["n_verified"]:3d} '
            f'time={out["wall_time_s"]:.2f}s'
        )
        results.append(out)
    return results, hit_deadline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solutions', required=True,
                    help='glob pattern for solutions_gpu_*.json')
    ap.add_argument('--geometry', required=True,
                    help='path to all_geometry_h11_<N>.json')
    ap.add_argument('--cy_index', type=int, required=True)
    ap.add_argument('--gamma', type=int, required=True)
    ap.add_argument('--time_limit', type=int, default=5,
                    help='per-K0 CP-SAT solver wall-time cap, seconds')
    ap.add_argument('--total_time_limit', type=int, default=None,
                    help='whole-run wall-time budget across ALL files and K0s, '
                         'seconds. Once exceeded, no new K0 is started and '
                         'whatever results we have are written out. The '
                         'currently-running K0 is not killed mid-solve; the '
                         'next K0 simply is not started.')
    ap.add_argument('--max_matrices', type=int, default=1000)
    ap.add_argument('--stability_mode', default='lazy',
                    choices=['lazy', 'eager', 'skip'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--m_bound', type=int, default=8,
                    help='domain cap on each K[i,a] (default 8). '
                         'When --tail_col_bound is set, this applies only to '
                         'columns 0..rank-2; the LAST column uses tail_col_bound.')
    ap.add_argument('--tail_col_bound', type=int, default=None,
                    help='|entry| bound on the LAST column of K (rank-1). '
                         'Defaults to --m_bound (uniform symmetric box). Set '
                         'wider (e.g. (rank-1)*m_bound) when K0s have unbounded '
                         'last col due to engine ignore_bounds=True.')
    ap.add_argument('--objective_cols', type=int, default=None,
                    help='Number of leading columns to include in the L1 '
                         'objective (cols 0..N-1). Defaults to all rank cols. '
                         'Set to rank-1 to make the last col a "free '
                         'consequence" of RowSum (Way 2 semantics).')
    ap.add_argument('--apply_column_lex', action='store_true',
                    help='break column-permutation symmetry via lex ordering '
                         '(may be slow in practice; off by default)')
    ap.add_argument('--output_dir', required=True,
                    help='where to write closed_*.json')
    args = ap.parse_args()

    # Resolve objective_cols (count) -> list of indices.
    objective_cols = (list(range(args.objective_cols))
                      if args.objective_cols is not None else None)

    paths = sorted(glob.glob(args.solutions))
    if not paths:
        print(f'no files match: {args.solutions}')
        sys.exit(1)
    print(f'Found {len(paths)} solutions file(s).')

    kappa, c2_tx, h11 = load_geometry(args.geometry, args.cy_index)
    print(f'Geometry: CY={args.cy_index} h11={h11} gamma={args.gamma} '
          f'c2_tx={c2_tx.tolist()}')

    run_start = time.time()
    deadline = (run_start + args.total_time_limit
                if args.total_time_limit is not None else None)
    if deadline is not None:
        print(f'Total-run deadline: {args.total_time_limit}s budget.')

    stopped_early = False
    for solutions_path in paths:
        if deadline is not None and time.time() >= deadline:
            print(f'[deadline reached; skipping remaining '
                  f'{len(paths) - paths.index(solutions_path)} file(s)]')
            stopped_early = True
            break
        stem = os.path.splitext(os.path.basename(solutions_path))[0]
        print(f'\n--- {os.path.basename(solutions_path)} ---')
        try:
            results, hit_deadline = process_file(
                solutions_path, kappa, c2_tx, args.gamma,
                time_limit=args.time_limit, max_matrices=args.max_matrices,
                stability_mode=args.stability_mode, workers=args.workers,
                m_bound=args.m_bound, apply_column_lex=args.apply_column_lex,
                deadline=deadline,
                tail_col_bound=args.tail_col_bound,
                objective_cols=objective_cols,
            )
        except Exception as e:
            print(f'  ERROR processing file: {e}')
            continue
        if hit_deadline:
            stopped_early = True

        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'closed_{stem}.json')
        payload = {
            'source': solutions_path,
            'cy_index': args.cy_index, 'gamma': args.gamma, 'h11': h11,
            'time_limit_s': args.time_limit,
            'total_time_limit_s': args.total_time_limit,
            'stopped_by_deadline': hit_deadline,
            'stability_mode': args.stability_mode,
            'm_bound': args.m_bound,
            'tail_col_bound': args.tail_col_bound,
            'objective_cols': args.objective_cols,
            'apply_column_lex': args.apply_column_lex,
            'results': results,
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f)
        n = len(results)
        n_pre = sum(1 for r in results if r['pre_all_pass'])
        n_post = sum(1 for r in results if r['post_all_pass'])
        n_feasible = sum(1 for r in results
                         if r['status'] in ('OPTIMAL', 'FEASIBLE'))
        n_verified_total = sum(r['n_verified'] for r in results)
        print(f'  --> {out_path}')
        print(f'  unique={n}  pre_8/8={n_pre}  solver_feasible={n_feasible}'
              f'  post_8/8={n_post}  verified_total={n_verified_total}')
        if stopped_early:
            break

    total_elapsed = time.time() - run_start
    if stopped_early:
        print(f'\n[Stopped early after {total_elapsed:.1f}s due to '
              f'--total_time_limit={args.total_time_limit}s]')
    else:
        print(f'\n[Finished all files in {total_elapsed:.1f}s]')


if __name__ == '__main__':
    main()
