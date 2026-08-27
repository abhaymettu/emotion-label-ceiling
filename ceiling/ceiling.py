"""Reliability-corrected ceiling for consensus-labelled emotion benchmarks.

Math and assumptions: ceiling/DERIVATION.md.

  .venv/bin/python ceiling/ceiling.py            # real data if present, else fixture
  .venv/bin/python ceiling/ceiling.py --check    # self-check on known-answer cases
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EMOTIONS, E_IDX, K, ROOT, count_matrix, load, majority_label, n_tied  # noqa: E402

OUT = ROOT / "ceiling" / "out"


# ---------------------------------------------------------------- reliability

def krippendorff_alpha_nominal(counts: np.ndarray) -> float:
    """Nominal Krippendorff's alpha from a units x categories count matrix.

    Units with fewer than 2 ratings contribute nothing, per Krippendorff.
    """
    c = counts[counts.sum(1) >= 2].astype(float)
    m = c.sum(1, keepdims=True)
    # coincidence matrix
    o = (c[:, :, None] * c[:, None, :]) / (m[:, :, None] - 1)
    o[:, np.arange(K), np.arange(K)] -= c / (m - 1)
    o = o.sum(0)
    nc = o.sum(1)
    n = nc.sum()
    d_o = o.sum() - np.trace(o)
    d_e = (nc.sum() ** 2 - (nc ** 2).sum()) / (n - 1)
    return float(1.0 - d_o / d_e)


def observed_agreement(counts: np.ndarray) -> float:
    """P(two randomly chosen raters of the same clip agree)."""
    c = counts[counts.sum(1) >= 2].astype(float)
    m = c.sum(1)
    return float((((c * (c - 1)).sum(1)) / (m * (m - 1))).mean())


# ------------------------------------------------------- dirichlet-multinomial

def _dm_nll(log_a: float, counts: np.ndarray, m: np.ndarray) -> float:
    a = np.exp(log_a)
    al = a * m                                  # rows x K
    n = counts.sum(1)
    s = al.sum(1)
    ll = (gammaln(s) - gammaln(n + s)
          + (gammaln(counts + al) - gammaln(al)).sum(1)).sum()
    return -ll


def fit_concentration(counts: np.ndarray, m: np.ndarray) -> float:
    """ML concentration `a` for counts_i ~ DirMult(R_i, a * m_i)."""
    r = minimize_scalar(_dm_nll, bounds=(-4.0, 8.0), method="bounded",
                        args=(counts, m), options={"xatol": 1e-4})
    return float(np.exp(r.x))


def class_profiles(counts: np.ndarray, intended: np.ndarray) -> np.ndarray:
    """Per-clip prior mean: the marginal response profile of its intended class."""
    m = np.empty((len(counts), K))
    for c in range(K):
        sel = intended == c
        if not sel.any():
            continue
        p = counts[sel].sum(0).astype(float)
        p = np.where(p > 0, p, 0.5)
        m[sel] = p / p.sum()
    return m


# ------------------------------------------------------------------- ceiling

def _tally(post, panel, rng, S):
    """S posterior-predictive draws of the consensus label. Returns clips x K tally."""
    tot = np.zeros_like(post)
    idx = np.arange(len(post))
    for _ in range(S):
        pi = rng.gamma(post)
        pi /= pi.sum(1, keepdims=True)
        tot[idx, majority_label(rng.multinomial(panel, pi), rng)] += 1
    return tot


def ceiling_bayes(counts, m, a, panel, rng, S=600) -> float:
    """E_X[max_y P(majority of a fresh R-rater panel = y | clip)].

    Monte Carlo over the Dirichlet posterior of pi_i and a fresh panel; see
    DERIVATION.md. Cross-fitted: half the draws choose the argmax, the other half
    estimate its probability. Without that split, max-of-six-noisy-estimates is
    itself upward-biased and the ceiling comes out too high for free.
    """
    post = counts + a * m
    h = max(S // 2, 50)
    A = _tally(post, panel, rng, h)
    B = _tally(post, panel, rng, h)
    star = A.argmax(1)
    return float((B[np.arange(len(B)), star] / B.sum(1)).mean())


def ceiling_plugin_naive(counts, panel, rng, S=600) -> float:
    """The optimistic estimator we do NOT use, kept to show the size of its bias.

    Empirical proportions treated as if they were the truth, no cross-fitting.
    """
    pi = counts / counts.sum(1, keepdims=True)
    tot = np.zeros_like(pi)
    idx = np.arange(len(pi))
    for _ in range(S):
        tot[idx, majority_label(rng.multinomial(panel, pi), rng)] += 1
    return float((tot / tot.sum(1, keepdims=True)).max(1).mean())


def split_half_oracle(df, rng, reps=20) -> tuple[float, float]:
    """Assumption-light lower-bound cross-check. Returns (mean, sd over reps)."""
    clip = df.clip_id.astype(str).to_numpy()
    resp = df.response_emotion.map(E_IDX).to_numpy()
    order = np.argsort(clip, kind="stable")
    clip, resp = clip[order], resp[order]
    starts = np.flatnonzero(np.r_[True, clip[1:] != clip[:-1]])
    ends = np.r_[starts[1:], len(clip)]
    keep = (ends - starts) >= 4                      # need >=2 per half
    starts, ends = starts[keep], ends[keep]

    accs = []
    for _ in range(reps):
        hits = tot = 0
        u = rng.random(len(clip))
        for s, e in zip(starts, ends):
            r, uu = resp[s:e], u[s:e]
            h = uu < np.median(uu)
            A, B = r[h], r[~h]
            if len(A) < 2 or len(B) < 2:
                continue
            ca = np.bincount(A, minlength=K)
            cb = np.bincount(B, minlength=K)
            pred = majority_label(ca[None, :], rng)[0]
            truth = majority_label(cb[None, :], rng)[0]
            hits += int(pred == truth)
            tot += 1
        accs.append(hits / tot)
    return float(np.mean(accs)), float(np.std(accs))


# -------------------------------------------------------------------- driver

def analyse(df, meta, panels, boot, rng) -> dict:
    counts, clips = count_matrix(df)
    per_clip = counts.sum(1)
    intended = (df.groupby("clip_id", observed=True).intended_emotion.first()
                .reindex(clips).map(E_IDX).to_numpy())
    m = class_profiles(counts, intended)
    a = fit_concentration(counts, m)
    R = int(np.median(per_clip))

    maj = majority_label(counts, rng)
    res = {
        **meta,
        "n_clips_used": int(len(counts)),
        "ratings_per_clip": {
            "median": R, "mean": float(per_clip.mean()),
            "min": int(per_clip.min()), "max": int(per_clip.max()),
        },
        "reliability": {
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(counts),
            "pairwise_percent_agreement": observed_agreement(counts),
            "n_clips": int((per_clip >= 2).sum()),
        },
        "dirichlet_concentration_a": a,
        "tie_rate_at_observed_panel": n_tied(counts) / len(counts),
        "consensus_vs_intended_accuracy": float((maj == intended).mean()),
        "panel_size_headline": R,
    }

    curve = {}
    for p in sorted(set(panels) | {R}):
        c = ceiling_bayes(counts, m, a, p, rng)
        curve[str(p)] = {"ceiling": c, "naive_plugin": ceiling_plugin_naive(counts, p, rng)}
    res["ceiling_by_panel_size"] = curve

    # cluster bootstrap over clips at the headline panel size
    bs = []
    for _ in range(boot):
        idx = rng.integers(0, len(counts), len(counts))
        bs.append(ceiling_bayes(counts[idx], m[idx], a, R, rng, S=300))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res["ceiling_headline"] = {
        "panel_size": R,
        "estimate": curve[str(R)]["ceiling"] if str(R) in curve else ceiling_bayes(counts, m, a, R, rng),
        "ci95": [float(lo), float(hi)],
        "n_bootstrap": boot,
        "n_clips": int(len(counts)),
    }

    sh_m, sh_s = split_half_oracle(df, rng)
    res["split_half_oracle"] = {
        "estimate": sh_m, "sd_over_reps": sh_s,
        "note": "conservative lower-bound cross-check; predictor and target both see R/2 raters",
    }

    # per-emotion ceiling, keyed on the intended class
    res["ceiling_by_intended_emotion"] = {}
    for c, e in enumerate(EMOTIONS):
        sel = intended == c
        if sel.sum() < 20:
            continue
        res["ceiling_by_intended_emotion"][e] = {
            "ceiling": ceiling_bayes(counts[sel], m[sel], a, R, rng),
            "consensus_vs_intended": float((maj[sel] == c).mean()),
            "n_clips": int(sel.sum()),
        }
    return res


def self_check() -> int:
    rng = np.random.default_rng(0)

    # alpha: perfect agreement is 1
    perfect = np.zeros((200, K), int)
    perfect[np.arange(200), rng.integers(0, K, 200)] = 8
    assert abs(krippendorff_alpha_nominal(perfect) - 1.0) < 1e-9, "alpha!=1 on perfect agreement"

    # alpha: independent random ratings sit at ~0
    rand = rng.multinomial(8, np.ones(K) / K, size=4000)
    assert abs(krippendorff_alpha_nominal(rand)) < 0.03, "alpha not ~0 on noise"

    # observed agreement on perfect data is 1
    assert abs(observed_agreement(perfect) - 1.0) < 1e-9

    # ceiling: a deterministic labelling process has ceiling 1
    assert ceiling_bayes(perfect * 4, np.ones((200, K)) / K, 0.01, 9, rng, 200) > 0.99, \
        "deterministic labels should give ceiling ~1"

    # ceiling: uniform-random labels give ceiling near chance-of-majority, not 1
    c_rand = ceiling_bayes(rand, np.ones((4000, K)) / K, 50.0, 7, rng, 200)
    assert c_rand < 0.45, f"ceiling on pure noise should be low, got {c_rand:.3f}"

    # ceiling must increase with panel size
    counts = rng.multinomial(7, [.45, .2, .12, .1, .08, .05], size=2000)
    m = np.tile([.45, .2, .12, .1, .08, .05], (2000, 1))
    a = fit_concentration(counts, m)
    seq = [ceiling_bayes(counts, m, a, p, rng, 400) for p in (3, 7, 21, 101)]
    assert all(x <= y + 0.01 for x, y in zip(seq, seq[1:])), f"ceiling not monotone in R: {seq}"
    assert seq[-1] > 0.97, f"ceiling should approach 1 as R grows, got {seq[-1]:.3f}"

    # naive plug-in is optimistic when clips are heterogeneous, which is the real case.
    # (With homogeneous clips the shrinkage target IS the truth and the sign flips --
    #  worth knowing: the correction only bites when there is between-clip spread.)
    het = rng.dirichlet(np.ones(K) * 1.5, size=2000)
    hc = np.array([rng.multinomial(7, q) for q in het])
    hm = np.tile(het.mean(0), (2000, 1))
    ha = fit_concentration(hc, hm)
    naive = ceiling_plugin_naive(hc, 7, rng)
    corrected = ceiling_bayes(hc, hm, ha, 7, rng, 400)
    assert naive > corrected + 0.05, (
        f"naive plug-in should be materially optimistic: naive={naive:.3f} "
        f"corrected={corrected:.3f}; if not, the correction is doing nothing")

    # concentration recovery
    p = rng.dirichlet(np.ones(K) * 3.0, size=3000)
    cc = np.array([rng.multinomial(12, q) for q in p])
    a_hat = fit_concentration(cc, np.ones((3000, K)) / K)
    assert 10 < a_hat < 30, f"concentration recovery off: got {a_hat:.2f}, truth 18"

    print("ok  alpha, ceiling monotonicity, plug-in bias, concentration recovery all pass")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--panels", default="1,3,5,7,9,11,15,21,31,51,101")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return self_check()

    rng = np.random.default_rng(a.seed)
    df, meta = load(a.ratings)
    panels = sorted({int(x) for x in a.panels.split(",")})
    res = {"by_modality": {}}
    res.update(analyse(df, meta, panels, a.bootstrap, rng))

    # modality matters: CREMA-D ran audio / visual / audiovisual separately
    for mod, sub in df.groupby("presented_modality", observed=True):
        if sub.clip_id.nunique() < 100:
            continue
        sm = {"source_file": meta["source_file"], "simulated": meta["simulated"],
              "n_ratings": len(sub), "n_clips": sub.clip_id.nunique(),
              "n_raters": sub.rater_id.nunique(), "n_actors": sub.actor_id.nunique(),
              "modalities": [mod]}
        res["by_modality"][str(mod)] = analyse(sub, sm, panels, a.bootstrap, rng)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ceiling.json").write_text(json.dumps(res, indent=2))

    tag = "  [SIMULATED]" if meta["simulated"] else ""
    h = res["ceiling_headline"]
    print(f"\nsource: {res['source_file']}{tag}")
    print(f"n = {res['n_ratings']:,} ratings, {res['n_clips']:,} clips, "
          f"{res['n_raters']:,} raters, median {res['panel_size_headline']} ratings/clip")
    print(f"Krippendorff alpha (nominal) = {res['reliability']['krippendorff_alpha_nominal']:.3f}")
    print(f"pairwise agreement           = {res['reliability']['pairwise_percent_agreement']:.3f}")
    print(f"consensus vs intended        = {res['consensus_vs_intended_accuracy']:.3f}")
    print(f"CEILING at R={h['panel_size']:<3d}            = {h['estimate']:.3f}  "
          f"95% CI [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}]  (n={h['n_clips']:,} clips)")
    print(f"split-half oracle (lower bd) = {res['split_half_oracle']['estimate']:.3f}")
    print(f"naive plug-in (biased, unused) = "
          f"{res['ceiling_by_panel_size'][str(h['panel_size'])]['naive_plugin']:.3f}")
    print(f"\nwrote {OUT / 'ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
