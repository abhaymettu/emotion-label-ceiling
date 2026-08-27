"""Bock's (1972) Nominal Response Model, fitted by marginal ML, and DIF on it.

This replaces the matched-decile crosstab in dif.py's `nominal_shift`, which was an
approximation to this and is kept alongside it for comparison (see invariance/README.md).

  .venv/bin/python invariance/nrm.py --check              # known-answer self-check
  .venv/bin/python invariance/nrm.py --modality audio     # primary analysis
  .venv/bin/python invariance/nrm.py --modality audio --perm 200 --loo

Model. Items i are the six intended emotions, categories k the six response buttons,
persons are raters, theta is a single latent rater trait.

    P(response = k | item i, theta) = exp(a_ik theta + c_ik) / sum_h exp(a_ih theta + c_ih)

Identification: a_i0 = c_i0 = 0 (reference category `anger`), reference group theta ~ N(0,1).
theta is oriented so the mean slope on the keyed (intended) category is positive, which
makes higher theta mean "more likely to answer as the actor was directed".

DIF is IRT-LR-DIF (Thissen, Steinberg & Wainer 1988/1993) adapted to nominal items:
all six items constrained equal across rater groups (focal group's theta mean and SD
free) is the baseline; freeing the studied item's 2(K-1)=10 parameters is the
alternative; 2*dLL is referred to chi2(10) AND to a rater-permutation null, because
the chi2 assumes conditional independence of trials given theta and this design has
unmodelled clip difficulty (see README).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ceiling"))
sys.path.insert(0, str(ROOT / "invariance"))
from common import EMOTIONS, E_IDX, K, load  # noqa: E402
from dif import add_groups, loo_consensus  # noqa: E402  (same groupings as LR-DIF)

OUT = ROOT / "invariance" / "out"
NODES = np.linspace(-4.0, 4.0, 61)


# --------------------------------------------------------------------- model

def _probs(A, C, x=NODES):
    """A, C: (I, K) -> category response probabilities (I, K, Q)."""
    z = A[:, :, None] * x[None, None, :] + C[:, :, None]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _prior(mu, sd, x=NODES):
    w = np.exp(-0.5 * ((x - mu) / sd) ** 2)
    return w / w.sum()


def _fit_item(N, x, a0, c0):
    """Weighted multinomial logit of response on the quadrature node.

    N: (K, Q) expected counts. Concave; L-BFGS with the analytic gradient.
    """
    nq = N.sum(axis=0)

    def obj(p):
        a = np.concatenate([[0.0], p[: K - 1]])
        c = np.concatenate([[0.0], p[K - 1:]])
        z = a[:, None] * x[None, :] + c[:, None]
        m = z.max(axis=0)
        lse = m + np.log(np.exp(z - m).sum(axis=0))
        ll = float((N * z).sum() - (nq * lse).sum())
        P = np.exp(z - lse)
        R = N - nq * P
        g = np.concatenate([(R * x[None, :]).sum(axis=1)[1:], R.sum(axis=1)[1:]])
        return -ll, -g

    r = minimize(obj, np.concatenate([a0[1:], c0[1:]]), jac=True, method="L-BFGS-B",
                 options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    a = np.concatenate([[0.0], r.x[: K - 1]])
    c = np.concatenate([[0.0], r.x[K - 1:]])
    return a, c


def fit_nrm(counts, group, free_item=None, init=None, maxit=400, tol=1e-7, x=NODES):
    """Marginal-ML (Bock-Aitkin EM) multigroup NRM.

    counts : (R, I, K) int, rater x item x response trial counts
    group  : (R,) int in {0, 1}. Group 0 is the reference (theta ~ N(0,1)).
    free_item : item index whose a/c differ by group, or None (fully anchored).

    Returns dict with A, C (G, I, K), mu, sd (G,), loglik, n_iter, converged.
    """
    R, I, _ = counts.shape
    G = 2
    if init is None:
        A = np.zeros((G, I, K))
        C = np.zeros((G, I, K))
        # sensible start: log marginal response shares, flat slopes
        sh = counts.sum(axis=0)
        sh = (sh + 1.0) / (sh + 1.0).sum(axis=1, keepdims=True)
        C[:] = np.log(sh) - np.log(sh[:, :1])
        A[:] = 0.3 * (np.eye(I, K) - 1.0 / K)
        mu = np.zeros(G)
        sd = np.ones(G)
    else:
        A, C, mu, sd = (init["A"].copy(), init["C"].copy(),
                        init["mu"].copy(), init["sd"].copy())
    if free_item is None:
        A[1] = A[0]
        C[1] = C[0]

    gmask = [group == g for g in range(G)]
    ng = np.array([m.sum() for m in gmask], float)
    ll_old = -np.inf
    it = 0
    conv = False
    for it in range(1, maxit + 1):
        # ---- E step
        post = np.empty((R, len(x)))
        ll = 0.0
        for g in range(G):
            m = gmask[g]
            if not m.any():
                continue
            lp = np.log(_probs(A[g], C[g], x))
            L = np.einsum("rik,ikq->rq", counts[m], lp)
            mx = L.max(axis=1, keepdims=True)
            w = np.exp(L - mx) * _prior(mu[g], sd[g], x)[None, :]
            s = w.sum(axis=1, keepdims=True)
            ll += float((np.log(s) + mx).sum())
            post[m] = w / s

        # ---- M step: expected counts (G, I, K, Q)
        N = np.stack([np.einsum("rq,rik->ikq", post[gmask[g]], counts[gmask[g]])
                      if gmask[g].any() else np.zeros((I, K, len(x)))
                      for g in range(G)])
        for i in range(I):
            if free_item is not None and i == free_item:
                for g in range(G):
                    A[g, i], C[g, i] = _fit_item(N[g, i], x, A[g, i], C[g, i])
            else:
                a, c = _fit_item(N[:, i].sum(axis=0), x, A[0, i], C[0, i])
                A[0, i] = A[1, i] = a
                C[0, i] = C[1, i] = c
        # focal group's latent distribution (reference held at 0, 1)
        for g in range(1, G):
            if not gmask[g].any():
                continue
            p = post[gmask[g]]
            mu[g] = float((p * x[None, :]).sum() / ng[g])
            v = float((p * (x[None, :] - mu[g]) ** 2).sum() / ng[g])
            sd[g] = float(np.sqrt(max(v, 1e-4)))

        if ll - ll_old < tol * max(1.0, abs(ll)):
            conv = True
            break
        ll_old = ll

    # orient theta so the keyed category loads positively
    keyed = np.array([A[0, i, i] for i in range(min(I, K))])
    if keyed.mean() < 0:
        A = -A
        mu = -mu
    return {"A": A, "C": C, "mu": mu, "sd": sd, "loglik": float(ll),
            "n_iter": it, "converged": bool(conv)}


# ----------------------------------------------------------------- DIF stats

def _mixture_weights(fit, ng, x=NODES):
    w = (ng[0] * _prior(fit["mu"][0], fit["sd"][0], x)
         + ng[1] * _prior(fit["mu"][1], fit["sd"][1], x))
    return w / w.sum()


def dif_stats(fit_free, item, ng, x=NODES):
    """Differential response functioning for one item, integrated over theta.

    dtvd is the total-variation distance between the two groups' six-category
    response distributions at equal theta, averaged over the population theta
    density. Directly comparable to the matched-decile TVD in dif.py.
    """
    w = _mixture_weights(fit_free, ng, x)
    PA = _probs(fit_free["A"][0], fit_free["C"][0], x)[item]   # (K, Q)
    PB = _probs(fit_free["A"][1], fit_free["C"][1], x)[item]
    d = PB - PA
    per_cat = (d * w[None, :]).sum(axis=1)
    j = int(np.abs(per_cat).argmax())
    return {
        "dtvd": float((0.5 * np.abs(d).sum(axis=0) * w).sum()),
        "dtvd_max_over_theta": float((0.5 * np.abs(d).sum(axis=0)).max()),
        "per_category_diff": {EMOTIONS[k]: float(per_cat[k]) for k in range(K)},
        "largest_shift_response": EMOTIONS[j],
        "largest_shift": float(per_cat[j]),
        "keyed_diff": float(per_cat[item]) if item < K else None,
    }


# ------------------------------------------------------------------ counting

def to_counts(df, group_col):
    """(counts (R,I,K), group (R,), rater_ids) for one grouping variable."""
    d = df[df[group_col].notna()]
    raters = np.sort(d.rater_id.unique())
    ri = pd.Series(np.arange(len(raters)), index=raters)
    counts = np.zeros((len(raters), K, K), dtype=np.int64)
    np.add.at(counts,
              (ri.loc[d.rater_id].to_numpy(),
               d.intended_emotion.map(E_IDX).to_numpy(),
               d.response_emotion.map(E_IDX).to_numpy()), 1)
    lv = sorted(d[group_col].dropna().unique())
    gmap = d.groupby("rater_id", observed=True)[group_col].first()
    group = (gmap.loc[raters].to_numpy() == lv[1]).astype(int)
    return counts, group, raters, [str(v) for v in lv]


# -------------------------------------------------------------------- driver

def analyse_group(counts, group, levels, perm=0, seed=0, verbose=True):
    ng = np.array([(group == 0).sum(), (group == 1).sum()], float)
    t0 = time.time()
    base = fit_nrm(counts, group)
    items = {}
    frees = {}
    for i in range(K):
        f = fit_nrm(counts, group, free_item=i, init=base)
        frees[i] = f
        lr = 2.0 * (f["loglik"] - base["loglik"])
        st = dif_stats(f, i, ng)
        from scipy.stats import chi2
        items[EMOTIONS[i]] = {
            "item": EMOTIONS[i],
            "n_trials": int(counts[:, i, :].sum()),
            "n_raters": int((counts[:, i, :].sum(axis=1) > 0).sum()),
            "n_raters_ref": int(ng[0]), "n_raters_focal": int(ng[1]),
            "lr_chi2": float(lr), "df": 2 * (K - 1),
            "p_chi2": float(chi2.sf(max(lr, 0.0), 2 * (K - 1))),
            **st,
        }
    res = {
        "levels": levels,
        "theta_focal_mean": float(base["mu"][1]), "theta_focal_sd": float(base["sd"][1]),
        "baseline_loglik": base["loglik"], "baseline_converged": base["converged"],
        "items": items, "fit_seconds": round(time.time() - t0, 2),
    }

    if perm:
        rng = np.random.default_rng(seed)
        null = {e: [] for e in EMOTIONS}
        nullc = {e: [] for e in EMOTIONS}
        for _ in range(perm):
            gp = rng.permutation(group)
            b = fit_nrm(counts, gp, init=base)
            for i in range(K):
                f = fit_nrm(counts, gp, free_item=i, init=b)
                null[EMOTIONS[i]].append(dif_stats(f, i, ng)["dtvd"])
                nullc[EMOTIONS[i]].append(2.0 * (f["loglik"] - b["loglik"]))
        for e in EMOTIONS:
            v = np.array(null[e]); c = np.array(nullc[e])
            obs = items[e]["dtvd"]
            items[e]["perm"] = {
                "n_permutations": perm,
                "null_dtvd_mean": float(v.mean()),
                "null_dtvd_q95": float(np.quantile(v, 0.95)),
                "p_perm_dtvd": float(((v >= obs).sum() + 1) / (perm + 1)),
                "null_chi2_mean": float(c.mean()),
                "p_perm_chi2": float(((c >= items[e]["lr_chi2"]).sum() + 1) / (perm + 1)),
                "excess_dtvd_over_null": float(obs - v.mean()),
            }
    return res, base, frees


def leave_one_rater_out(counts, group, base, obs, max_raters=None, seed=0, verbose=True):
    """Refit the whole DIF analysis with each rater deleted, warm-started.

    Reports, per item, the min/max dtvd over deletions and the single most
    influential rater. Jackknife SE is also reported (n-1 scaling).
    """
    R = counts.shape[0]
    idx = np.arange(R)
    rng = np.random.default_rng(seed)
    if max_raters and max_raters < R:
        idx = np.sort(rng.choice(R, max_raters, replace=False))
    vals = {e: np.empty(len(idx)) for e in EMOTIONS}
    t0 = time.time()
    for n, r in enumerate(idx):
        keep = np.ones(R, bool)
        keep[r] = False
        c2, g2 = counts[keep], group[keep]
        ng2 = np.array([(g2 == 0).sum(), (g2 == 1).sum()], float)
        b = fit_nrm(c2, g2, init=base, maxit=60)
        for i in range(K):
            f = fit_nrm(c2, g2, free_item=i, init=b, maxit=60)
            vals[EMOTIONS[i]][n] = dif_stats(f, i, ng2)["dtvd"]
        if verbose and (n + 1) % 200 == 0:
            el = time.time() - t0
            print(f"    loo {n+1}/{len(idx)}  {el:.0f}s "
                  f"(eta {el/(n+1)*(len(idx)-n-1):.0f}s)", file=sys.stderr)
    out = {}
    for e in EMOTIONS:
        v = vals[e]
        m = v.mean()
        # delete-one jackknife SE, exact when every rater is deleted
        se = float(np.sqrt((len(v) - 1) / len(v) * ((v - m) ** 2).sum()))
        out[e] = {
            "n_deletions": int(len(idx)), "exhaustive": bool(len(idx) == R),
            "observed_dtvd": float(obs[e]),
            "loo_min": float(v.min()), "loo_max": float(v.max()),
            "loo_mean": float(m),
            "max_abs_change": float(np.abs(v - obs[e]).max()),
            "jackknife_se": se if len(idx) == R else None,
        }
    return out


# ---------------------------------------------------------------- self-check

def _simulate(rng, n_r=1500, n_rep=5, dif_item=2, dif_size=1.2, groups=True):
    """Generate from a known NRM. Group 1 gets a planted shift on `dif_item`:
    intercept mass moved from the keyed category onto `neutral`."""
    A = 0.9 * (np.eye(K) - 1.0 / K) + 0.15 * rng.standard_normal((K, K))
    C = -0.8 * np.ones((K, K)) + 1.6 * np.eye(K) + 0.2 * rng.standard_normal((K, K))
    A -= A[:, :1]
    C -= C[:, :1]
    A2, C2 = A.copy(), C.copy()
    if groups and dif_item is not None:
        C2[dif_item, E_IDX["neutral"]] += dif_size
        C2[dif_item, dif_item] -= dif_size
    grp = rng.integers(0, 2, n_r) if groups else np.zeros(n_r, int)
    th = rng.standard_normal(n_r)
    counts = np.zeros((n_r, K, K), dtype=np.int64)
    for r in range(n_r):
        a, c = (A2, C2) if grp[r] == 1 else (A, C)
        z = a * th[r] + c
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        for i in range(K):
            counts[r, i] = rng.multinomial(n_rep, p[i])
    return counts, grp, A, C, A2, C2


def self_check() -> int:
    rng = np.random.default_rng(0)

    # 1. parameter recovery, single group, no DIF
    counts, grp, A, C, *_ = _simulate(rng, n_r=2000, n_rep=6, groups=False)
    f = fit_nrm(counts, np.zeros(len(counts), int))
    ea = np.abs(f["A"][0] - A).max()
    ec = np.abs(f["C"][0] - C).max()
    assert f["converged"], "EM did not converge on clean data"
    assert ea < 0.25, f"slope recovery off by {ea:.3f}"
    assert ec < 0.25, f"intercept recovery off by {ec:.3f}"

    # 2. planted nominal DIF on item 'fear', keyed mass moved onto 'neutral'
    counts, grp, A, C, A2, C2 = _simulate(rng, n_r=1800, n_rep=6,
                                          dif_item=E_IDX["fear"], dif_size=1.2)
    ng = np.array([(grp == 0).sum(), (grp == 1).sum()], float)
    base = fit_nrm(counts, grp)
    hit = fit_nrm(counts, grp, free_item=E_IDX["fear"], init=base)
    st = dif_stats(hit, E_IDX["fear"], ng)
    lr = 2 * (hit["loglik"] - base["loglik"])
    assert lr > 50, f"planted DIF missed: chi2={lr:.1f}"
    assert st["largest_shift_response"] == "neutral", \
        f"wrong distractor named: {st['largest_shift_response']}"
    assert st["largest_shift"] > 0.05, f"planted shift too small: {st['largest_shift']:.3f}"
    assert st["keyed_diff"] < 0, "planted DIF has the wrong sign on the keyed category"

    # 3. no false positives on the five clean items
    worst = 0.0
    for i in range(K):
        if i == E_IDX["fear"]:
            continue
        f2 = fit_nrm(counts, grp, free_item=i, init=base)
        worst = max(worst, dif_stats(f2, i, ng)["dtvd"])
    assert worst < st["dtvd"] / 3, \
        f"clean item dtvd={worst:.4f} too close to planted {st['dtvd']:.4f}"

    # 4. permutation null sits below the observed effect
    rngp = np.random.default_rng(1)
    nulls = []
    for _ in range(12):
        gp = rngp.permutation(grp)
        b = fit_nrm(counts, gp, init=base)
        f3 = fit_nrm(counts, gp, free_item=E_IDX["fear"], init=b)
        nulls.append(dif_stats(f3, E_IDX["fear"], ng)["dtvd"])
    assert max(nulls) < st["dtvd"], \
        f"permutation null {max(nulls):.4f} reaches observed {st['dtvd']:.4f}"

    print(f"ok  NRM recovers params (max err a={ea:.3f} c={ec:.3f}); "
          f"planted nominal DIF found (chi2={lr:.0f}, dtvd={st['dtvd']:.3f}, "
          f"distractor={st['largest_shift_response']}); clean items max dtvd={worst:.4f}; "
          f"permutation null max={max(nulls):.4f}")
    return 0


def mirt_check(n_r=4000) -> int:
    """Independent-implementation check: fit the same NRM here and in R's mirt.

    Balanced one-response-per-item fixture (the classic NRM design mirt expects),
    then compare fitted category response curves. mirt's default nominal itemtype
    is a *restricted* NRM; mirt_check.R frees the top-category slope so both fit
    Bock's general model. Skipped with a clear message if Rscript/mirt are absent.
    """
    import shutil
    import subprocess

    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "mirt_fixture.csv"
    rng = np.random.default_rng(7)
    counts, _, A, C, *_ = _simulate(rng, n_r=n_r, n_rep=1, groups=False)
    pd.DataFrame(counts.argmax(axis=2), columns=EMOTIONS).to_csv(csv, index=False)
    f = fit_nrm(counts, np.zeros(n_r, int), maxit=800, tol=1e-9)

    if not shutil.which("Rscript"):
        print("Rscript not found; cross-check skipped (not a failure, but the "
              "independent-implementation check did not run)", file=sys.stderr)
        return 0
    r = subprocess.run(["Rscript", str(ROOT / "invariance" / "mirt_check.R"), str(csv)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("mirt cross-check could not run (is the R package `mirt` installed?)",
              file=sys.stderr)
        return 0
    ll_mirt = float(r.stdout.split()[2])
    x = np.linspace(-4, 4, 61)
    mine = _probs(f["A"][0], f["C"][0], x)
    theirs = (pd.read_csv(str(csv).replace(".csv", "_mirt_probs.csv"))
              .to_numpy().reshape(len(x), K, K).transpose(1, 2, 0))
    w = np.exp(-0.5 * x ** 2)
    w /= w.sum()
    d = np.abs(mine - theirs)
    res = {
        "n_persons": n_r, "n_items": K, "n_categories": K,
        "loglik_nrm_py": round(f["loglik"], 3), "loglik_mirt": round(ll_mirt, 3),
        "loglik_gap": round(f["loglik"] - ll_mirt, 3),
        "max_abs_prob_diff_all_theta": round(float(d.max()), 6),
        "max_abs_prob_diff_core_theta_2.5": round(float(d[:, :, np.abs(x) <= 2.5].max()), 6),
        "density_weighted_mean_abs_prob_diff": round(float((d * w).sum() / (K * K)), 8),
    }
    (OUT / "nrm-mirt-crosscheck.json").write_text(json.dumps(res, indent=2))
    assert abs(res["loglik_gap"]) < 1.0, f"loglik disagrees with mirt: {res}"
    assert res["max_abs_prob_diff_all_theta"] < 0.05, f"curves disagree with mirt: {res}"
    print("ok  matches R/mirt: loglik " f"{res['loglik_nrm_py']} vs {res['loglik_mirt']}, "
          f"max |dP| = {res['max_abs_prob_diff_all_theta']:.4f} "
          f"(density-weighted mean {res['density_weighted_mean_abs_prob_diff']:.2e})")
    return 0


# ------------------------------------------------------------------- runner

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--modality", default=None, help="audio | visual | audiovisual")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=0, help="rater-permutation replicates")
    ap.add_argument("--loo", action="store_true", help="leave-one-rater-out refits")
    ap.add_argument("--loo-groups", default="grp_style_resid",
                    help="comma-separated grouping vars to run LOO on")
    ap.add_argument("--loo-max", type=int, default=0, help="0 = every rater")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--key", default="intent", choices=["intent", "consensus_loo"],
                    help="define items by the actor's intent (default) or by the "
                         "leave-one-rater-out crowd majority label")
    ap.add_argument("--mirt-check", action="store_true",
                    help="fit the same NRM in R/mirt and compare (needs Rscript + mirt)")
    a = ap.parse_args()
    if a.check:
        return self_check()
    if a.mirt_check:
        return mirt_check()

    rng = np.random.default_rng(a.seed)
    df, meta = load(a.ratings)
    if a.modality:
        df = df[df.presented_modality == a.modality]
        meta = {**meta, "modality_filter": a.modality, "n_ratings": int(len(df))}
    df = add_groups(df, rng)
    if a.key == "consensus_loo":
        df["intended_emotion"] = loo_consensus(df, rng)
        df = df[df.intended_emotion.notna()].copy()
    groups = [c for c in df.columns if c.startswith("grp_")]

    res = {**meta, "model": "Bock (1972) nominal response model, multigroup MML-EM",
           "items": {"intent": "six intended emotions",
                     "consensus_loo": "six leave-one-rater-out crowd majority "
                                      "classes"}[a.key],
           "item_key": a.key, "categories": EMOTIONS,
           "reference_category": EMOTIONS[0],
           "quadrature_nodes": len(NODES),
           "dif_test": "IRT-LR-DIF, all other items anchored, df=2(K-1)=10",
           "effect_size": "dtvd = TVD between groups' six-category response "
                          "distributions at equal theta, averaged over the population "
                          "theta density",
           "grouping_variables": groups, "by_group": {}}

    loo_groups = [g.strip() for g in a.loo_groups.split(",") if g.strip()]
    rows = []
    for gcol in groups:
        d = df[df._style_holdout] if gcol.startswith("grp_style") else df
        counts, group, raters, levels = to_counts(d, gcol)
        print(f"\n=== {gcol}  {levels}  "
              f"raters={len(raters):,} trials={counts.sum():,}", file=sys.stderr)
        r, base, frees = analyse_group(counts, group, levels, perm=a.perm, seed=a.seed)
        if a.loo and gcol in loo_groups:
            obs = {e: r["items"][e]["dtvd"] for e in EMOTIONS}
            r["leave_one_rater_out"] = leave_one_rater_out(
                counts, group, base, obs, max_raters=a.loo_max or None, seed=a.seed)
        res["by_group"][gcol] = r
        for e in EMOTIONS:
            it = r["items"][e]
            rows.append({
                "group_var": gcol, "item": e, "levels": "/".join(levels),
                "n_trials": it["n_trials"], "n_raters": it["n_raters"],
                "dtvd_nrm": round(it["dtvd"], 5),
                "lr_chi2": round(it["lr_chi2"], 2), "df": it["df"],
                "p_chi2": it["p_chi2"],
                "largest_shift_response": it["largest_shift_response"],
                "largest_shift": round(it["largest_shift"], 4),
                "keyed_diff": round(it["keyed_diff"], 4),
                "p_perm_dtvd": it.get("perm", {}).get("p_perm_dtvd"),
                "null_dtvd_mean": (round(it["perm"]["null_dtvd_mean"], 5)
                                   if "perm" in it else None),
                "excess_over_null": (round(it["perm"]["excess_dtvd_over_null"], 5)
                                     if "perm" in it else None),
                "loo_min": (round(r["leave_one_rater_out"][e]["loo_min"], 5)
                            if "leave_one_rater_out" in r else None),
                "loo_max": (round(r["leave_one_rater_out"][e]["loo_max"], 5)
                            if "leave_one_rater_out" in r else None),
            })

    tab = pd.DataFrame(rows)
    rank = tab.groupby("item").dtvd_nrm.mean().sort_values(ascending=False)
    res["least_invariant_items"] = [{"item": k, "mean_dtvd": float(v)}
                                    for k, v in rank.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{a.modality}" if a.modality else ""
    suffix += "-consensusloo" if a.key == "consensus_loo" else ""
    tab.to_csv(OUT / f"nrm-dif{suffix}.csv", index=False)
    (OUT / f"nrm-dif{suffix}.json").write_text(json.dumps(res, indent=2))

    tag = "  [SIMULATED]" if meta["simulated"] else ""
    print(f"\nsource: {res['source_file']}{tag}   modality={a.modality or 'all pooled'}"
          f"   n={res['n_ratings']:,} trials, {res['n_raters']:,} raters")
    show = [c for c in ["group_var", "item", "n_trials", "dtvd_nrm", "lr_chi2",
                        "p_chi2", "largest_shift_response", "largest_shift",
                        "keyed_diff", "p_perm_dtvd", "loo_min", "loo_max"]
            if tab[c].notna().any()]
    print(tab.sort_values(["group_var", "dtvd_nrm"], ascending=[True, False])[show]
          .to_string(index=False))
    print("\nleast invariant items (mean dtvd across grouping variables):")
    for r in res["least_invariant_items"]:
        print(f"  {r['item']:9s} {r['mean_dtvd']:.4f}")
    print(f"\nwrote {OUT / f'nrm-dif{suffix}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
