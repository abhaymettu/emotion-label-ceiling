"""Nominal inter-rater reliability, computed from unit-by-category count matrices.

Everything downstream works off one object: a (units x categories) integer
matrix N, where N[u, c] is how many raters gave unit u category c. That form
handles CREMA-D's variable rater counts (6-12 per clip-modality cell) without
ever materializing a 2443 x 22326 rater-by-unit matrix.

Run this file directly for the self-checks (validates alpha against the
`krippendorff` package and kappa against `statsmodels`).
"""

import numpy as np

RNG = np.random.default_rng(20260826)


def counts_matrix(df, unit_cols, value_col, categories):
    """(units x categories) counts, plus the unit keys in row order."""
    g = df.groupby(unit_cols + [value_col], observed=True).size().unstack(value_col, fill_value=0)
    g = g.reindex(columns=list(categories), fill_value=0)
    return g.to_numpy(dtype=np.int64), g.index


def krippendorff_alpha(N):
    """Krippendorff's alpha, nominal metric, from a unit-by-category count matrix.

    Coincidence-matrix form. Units with fewer than 2 ratings carry no
    information about disagreement and are dropped, exactly as Krippendorff
    specifies. Returns nan if fewer than 2 units survive.
    """
    N = np.asarray(N, dtype=np.float64)
    m = N.sum(axis=1)
    N = N[m >= 2]
    m = m[m >= 2]
    if len(N) < 2:
        return float("nan")
    n = m.sum()
    # observed disagreement
    d_o = float((((m ** 2) - (N ** 2).sum(axis=1)) / (m - 1)).sum() / n)
    # expected disagreement from the pooled category marginals
    n_c = N.sum(axis=0)
    d_e = float((n ** 2 - (n_c ** 2).sum()) / (n * (n - 1)))
    if d_e == 0:
        return float("nan")
    return 1.0 - d_o / d_e


def alpha_fixed_marginals(N, n_c):
    """Alpha for a subset of units, but with D_e taken from a wider marginal.

    Class-wise alpha is otherwise not comparable across classes: restricting
    units to one intended emotion also restricts the response marginals, which
    shrinks expected disagreement and mechanically drags alpha down. Holding the
    marginals fixed at the full-sample distribution asks a different and more
    answerable question -- how much of the *overall* expected disagreement this
    subset resolves. Report both; neither alone is the answer.
    """
    N = np.asarray(N, dtype=np.float64)
    m = N.sum(axis=1)
    N, m = N[m >= 2], m[m >= 2]
    if len(N) < 2:
        return float("nan")
    d_o = float((((m ** 2) - (N ** 2).sum(axis=1)) / (m - 1)).sum() / m.sum())
    n_c = np.asarray(n_c, dtype=np.float64)
    n = n_c.sum()
    d_e = float((n ** 2 - (n_c ** 2).sum()) / (n * (n - 1)))
    return 1.0 - d_o / d_e if d_e else float("nan")


def fleiss_kappa(N):
    """Fleiss' kappa. Requires every unit to have the same number of raters.

    Raises if the design is unbalanced -- that is the point of reporting it
    alongside alpha, and silently averaging over a ragged design would hide it.
    """
    N = np.asarray(N, dtype=np.float64)
    m = N.sum(axis=1)
    if len(np.unique(m)) != 1:
        raise ValueError(f"Fleiss' kappa needs a constant rater count; got {sorted(set(m.tolist()))}")
    m = m[0]
    if m < 2:
        return float("nan")
    n_units = len(N)
    p_bar = float((((N ** 2).sum(axis=1) - m) / (m * (m - 1))).mean())
    p_c = N.sum(axis=0) / (n_units * m)
    p_e = float((p_c ** 2).sum())
    return (p_bar - p_e) / (1 - p_e) if p_e != 1 else float("nan")


def bootstrap_ci(N, stat=krippendorff_alpha, reps=2000, alpha_level=0.05, rng=RNG):
    """Percentile CI by resampling units with replacement.

    Units, not individual ratings: ratings within a clip are not independent,
    which is the entire thing being measured.
    """
    N = np.asarray(N)
    n = len(N)
    if n < 2:
        return (float("nan"), float("nan"), 0)
    draws = np.empty(reps)
    for i in range(reps):
        draws[i] = stat(N[rng.integers(0, n, n)])
    lo, hi = np.nanpercentile(draws, [100 * alpha_level / 2, 100 * (1 - alpha_level / 2)])
    return float(lo), float(hi), int(np.isfinite(draws).sum())


def coincidence_matrix(N):
    """Krippendorff coincidence matrix: how often category c and k land on the same unit.

    Symmetric, and this -- not the response-vs-intended confusion matrix -- is
    the honest answer to "which emotion pairs do raters conflate", because it
    never mentions the actor's intent.
    """
    N = np.asarray(N, dtype=np.float64)
    m = N.sum(axis=1)
    N, m = N[m >= 2], m[m >= 2]
    w = 1.0 / (m - 1)
    o = (N * w[:, None]).T @ N
    np.fill_diagonal(o, np.diag(o) - (N * w[:, None]).sum(axis=0))
    return o


def _selfcheck():
    import krippendorff as kd
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss

    rng = np.random.default_rng(0)
    cats = 6

    # alpha against the reference implementation, on ragged and balanced designs
    for m_lo, m_hi in [(6, 13), (10, 11), (2, 4)]:
        for units in (60, 400):
            N = np.zeros((units, cats), dtype=int)
            for u in range(units):
                m = rng.integers(m_lo, m_hi)
                p = rng.dirichlet(np.full(cats, 0.7))
                N[u] = rng.multinomial(m, p)
            # reference wants a (coders x units) matrix with nan padding
            m_max = int(N.sum(axis=1).max())
            wide = np.full((m_max, len(N)), np.nan)
            for u, row in enumerate(N):
                vals = np.repeat(np.arange(cats), row)
                wide[: len(vals), u] = vals
            ref = kd.alpha(reliability_data=wide, level_of_measurement="nominal")
            ours = krippendorff_alpha(N)
            assert abs(ours - ref) < 1e-9, (m_lo, units, ours, ref)

    # kappa against statsmodels, and it must refuse a ragged design
    N = np.array([rng.multinomial(10, rng.dirichlet(np.full(cats, 0.7))) for _ in range(300)])
    assert abs(fleiss_kappa(N) - sm_fleiss(N)) < 1e-9
    try:
        fleiss_kappa(np.array([[5, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0]]))
        raise AssertionError("fleiss_kappa accepted an unbalanced design")
    except ValueError:
        pass

    # degenerate cases
    assert krippendorff_alpha(np.array([[10, 0, 0, 0, 0, 0]] * 50)) != krippendorff_alpha(np.zeros((2, 6)))
    perfect = np.zeros((60, cats), dtype=int)
    for u in range(60):
        perfect[u, u % cats] = 8
    assert abs(krippendorff_alpha(perfect) - 1.0) < 1e-12

    # coincidence matrix: symmetric, and its off-diagonal mass reproduces D_o
    N = np.array([rng.multinomial(int(rng.integers(6, 13)), rng.dirichlet(np.full(cats, 0.7)))
                  for _ in range(500)])
    o = coincidence_matrix(N)
    assert np.allclose(o, o.T)
    m = N.sum(axis=1)
    d_o = float((((m ** 2) - (N ** 2).sum(axis=1)) / (m - 1)).sum() / m.sum())
    assert abs((o.sum() - np.trace(o)) / m.sum() - d_o) < 1e-9

    # alpha_fixed_marginals reduces to plain alpha when the marginals are the subset's own
    assert abs(alpha_fixed_marginals(N, N.sum(axis=0)) - krippendorff_alpha(N)) < 1e-12
    print("agreement selfcheck: OK")


if __name__ == "__main__":
    _selfcheck()
