"""Differential item functioning across annotators, dichotomous (correct/incorrect).

Rationale and model justification: invariance/METHOD.md. Results and honest limits:
invariance/README.md.

The nominal (which-wrong-answer) question is answered by a fitted Bock nominal
response model in invariance/nrm.py. `nominal_shift` below is the matched-decile
approximation to it, kept for comparison and labelled as such.

  .venv/bin/python invariance/dif.py           # real data if present, else fixture
  .venv/bin/python invariance/dif.py --check   # known-answer self-check
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ceiling"))
from common import EMOTIONS, E_IDX, K, load  # noqa: E402

OUT = ROOT / "invariance" / "out"
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="statsmodels")

# Jodoin & Gierl (2001) is the one to read; Zumbo & Thomas (1997) is too lenient.
JG = [(0.035, "negligible"), (0.070, "moderate"), (np.inf, "large")]
ZT = [(0.130, "negligible"), (0.260, "moderate"), (np.inf, "large")]


def _band(v, table):
    return next(lab for thr, lab in table if v < thr)


# ------------------------------------------------------------------- grouping

def add_groups(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Attach the rater grouping variables described in METHOD.md."""
    df = df.copy()
    df["correct"] = (df.response_emotion.to_numpy() == df.intended_emotion.to_numpy()).astype(int)
    g = df.groupby("rater_id", observed=True)

    if "response_time_ms" in df:
        rt = g.response_time_ms.median()
        df["grp_speed"] = df.rater_id.map(
            dict(zip(rt.index, np.where(rt > rt.median(), "deliberate", "fast"))))
    if "response_intensity" in df and df.response_intensity.notna().any():
        ex = g.response_intensity.apply(lambda s: (s - 50).abs().mean())
        df["grp_extremity"] = df.rater_id.map(
            dict(zip(ex.index, np.where(ex > ex.median(), "extreme", "moderate"))))
    if "question_num" in df:
        qm = g.question_num.transform("median")
        df["grp_position"] = np.where(df.question_num > qm, "late", "early")

    # style: PC1 of the rater confusion profile, fitted on a held-out half so the
    # grouping is not read off the same trials the DIF test uses.
    half = rng.random(len(df)) < 0.5
    prof = (df[half].groupby(["rater_id", "intended_emotion", "response_emotion"],
                             observed=True).size().unstack(fill_value=0))
    prof = prof.div(prof.sum(axis=1).clip(lower=1), axis=0).unstack(fill_value=0.0)
    Xp = np.array(prof.to_numpy(float), copy=True)
    Xp -= Xp.mean(0)
    pc1 = np.linalg.svd(Xp, full_matrices=False)[2][0]
    sc = pd.Series(Xp @ pc1, index=prof.index)
    df["grp_style"] = df.rater_id.map(
        dict(zip(sc.index, np.where(sc > sc.median(), "style_B", "style_A"))))

    # A confusion profile is partly just accuracy, so PC1 may be measuring ability
    # rather than construal. Residualise it on the rater's own accuracy (also from
    # the PCA half only) and split on that: "different confusions at equal ability".
    acc = df[half].groupby("rater_id", observed=True).correct.mean().reindex(sc.index)
    ok = acc.notna()
    b = np.polyfit(acc[ok], sc[ok], 1)
    resid = sc[ok] - np.polyval(b, acc[ok])
    df.attrs["style_ability_corr"] = float(np.corrcoef(acc[ok], sc[ok])[0, 1])
    df["grp_style_resid"] = df.rater_id.map(
        dict(zip(resid.index, np.where(resid > resid.median(), "resid_B", "resid_A"))))

    df["_style_holdout"] = ~half        # test DIF on the trials PCA did not see
    return df


def loo_consensus(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Leave-one-rater-out crowd consensus label for each rating row.

    The clip's majority response computed from every rater EXCEPT the one whose row
    it is, within the same presented modality. Ties broken uniformly at random.

    Why it exists: keying DIF on the ordinary crowd consensus is circular, because the
    rater under test helped build the label they are scored against. Dropping their own
    vote removes that particular circularity (it does not remove the fact that the
    remaining ~9 raters are drawn from the same pool).
    """
    key = list(zip(df.presented_modality.astype(str), df.clip_id.astype(str)))
    kk = pd.Index(pd.unique(pd.Series(key)))
    ki = pd.Series(np.arange(len(kk)), index=kk).loc[key].to_numpy()
    ri = df.response_emotion.map(E_IDX).to_numpy()
    tot = np.zeros((len(kk), K), np.int64)
    np.add.at(tot, (ki, ri), 1)
    own = tot[ki].copy()
    own[np.arange(len(ri)), ri] -= 1                 # drop this rater's own vote
    mx = own.max(axis=1, keepdims=True)
    pick = (rng.random(own.shape) * (own == mx)).argmax(axis=1)
    out = pd.Series(np.array(EMOTIONS)[pick], index=df.index)
    out[own.sum(axis=1) == 0] = pd.NA                # nobody else rated the clip
    return out


def rest_scores(df: pd.DataFrame) -> pd.Series:
    """Rater ability on the other five items. Matching on the total score would let
    the studied item contaminate its own matching criterion."""
    tot = df.groupby("rater_id", observed=True).correct.transform("sum")
    n = df.groupby("rater_id", observed=True).correct.transform("size")
    itot = df.groupby(["rater_id", "intended_emotion"], observed=True).correct.transform("sum")
    inn = df.groupby(["rater_id", "intended_emotion"], observed=True).correct.transform("size")
    return (tot - itot) / (n - inn).clip(lower=1)


# ------------------------------------------------------------------------ DIF

def _nagelkerke(res, n: int) -> float:
    ll0 = res.llnull
    cs = 1.0 - np.exp(2.0 * (ll0 - res.llf) / n)
    return float(cs / (1.0 - np.exp(2.0 * ll0 / n)))


def lr_dif(sub: pd.DataFrame, group_col: str) -> dict | None:
    """Swaminathan & Rogers logistic-regression DIF for one item."""
    lv = sorted(sub[group_col].dropna().unique())
    if len(lv) != 2 or len(sub) < 200:
        return None
    y = sub.correct.to_numpy(float)
    theta = sub.rest.to_numpy(float)
    theta = (theta - theta.mean()) / (theta.std() or 1.0)
    g = (sub[group_col].to_numpy() == lv[1]).astype(float)
    n = len(y)
    cl = sub.rater_id.to_numpy()

    def fit(X, robust: bool):
        m = sm.Logit(y, sm.add_constant(X, has_constant="add"))
        r = m.fit(disp=0, maxiter=200)
        if robust:
            r = m.fit(disp=0, maxiter=200, cov_type="cluster",
                      cov_kwds={"groups": cl, "use_correction": True})
        return r

    r0 = fit(theta[:, None], False)
    r2 = fit(np.column_stack([theta, g, theta * g]), False)
    r2r = fit(np.column_stack([theta, g, theta * g]), True)

    d_r2 = _nagelkerke(r2, n) - _nagelkerke(r0, n)
    # base rates, unmatched, for readability
    p0 = float(y[g == 0].mean())
    p1 = float(y[g == 1].mean())
    return {
        "group_var": group_col, "levels": [str(lv[0]), str(lv[1])],
        "n_trials": int(n), "n_raters": int(sub.rater_id.nunique()),
        "delta_r2_nagelkerke": d_r2,
        "jodoin_gierl": _band(d_r2, JG), "zumbo_thomas": _band(d_r2, ZT),
        "beta_group_uniform": float(r2r.params[2]),
        "z_group_clustered": float(r2r.tvalues[2]),
        "p_group_clustered": float(r2r.pvalues[2]),
        "beta_interaction_nonuniform": float(r2r.params[3]),
        "z_interaction_clustered": float(r2r.tvalues[3]),
        "p_interaction_clustered": float(r2r.pvalues[3]),
        "raw_accuracy": {str(lv[0]): p0, str(lv[1]): p1, "diff": p1 - p0},
    }


def mantel_haenszel(sub: pd.DataFrame, group_col: str, bins: int = 10) -> dict | None:
    """MH odds ratio with ETS A/B/C classification. Uniform DIF only."""
    lv = sorted(sub[group_col].dropna().unique())
    if len(lv) != 2:
        return None
    q = pd.qcut(sub.rest, bins, duplicates="drop", labels=False)
    g = (sub[group_col].to_numpy() == lv[1]).astype(int)
    y = sub.correct.to_numpy(int)
    num = den = 0.0
    for b in np.unique(q[~pd.isna(q)]):
        s = q == b
        a = ((g[s] == 0) & (y[s] == 1)).sum(); bb = ((g[s] == 0) & (y[s] == 0)).sum()
        c = ((g[s] == 1) & (y[s] == 1)).sum(); d = ((g[s] == 1) & (y[s] == 0)).sum()
        t = a + bb + c + d
        if t == 0:
            continue
        num += a * d / t
        den += bb * c / t
    if den <= 0 or num <= 0:
        return None
    a_mh = num / den
    delta = -2.35 * np.log(a_mh)
    ets = "A (negligible)" if abs(delta) < 1.0 else (
        "B (moderate)" if abs(delta) < 1.5 else "C (large)")
    return {"alpha_mh": float(a_mh), "delta_mh": float(delta), "ets": ets}


def nominal_shift(sub: pd.DataFrame, group_col: str) -> dict | None:
    """APPROXIMATION to a nominal response model. Not a fitted NRM.

    Which wrong answer, not just whether wrong: dichotomised DIF cannot see 'group A
    hears fear as sadness while group B hears it as fear'. This gets at that by
    stratifying on observed rest score in deciles and averaging the by-group response
    distribution over strata.

    It is an approximation in three specific ways, all of which the fitted model in
    invariance/nrm.py avoids:

      1. It matches on the *observed* rest score, which is a noisy proxy for latent
         ability, so matching is imperfect and the effect is attenuated.
      2. Ten strata is a coarse discretisation of a continuous trait.
      3. It has no null. Ten-decile crosstabs of six categories on finite cells always
         return a nonzero total-variation distance, so the number has no zero point
         and cannot be tested.

    Kept, and reported next to the NRM result, because a reader should be able to see
    whether the cheap approximation and the fitted model agree. Column name in the CSV
    is `decile_tvd_approx` for exactly that reason.
    """
    lv = sorted(sub[group_col].dropna().unique())
    if len(lv) != 2:
        return None
    q = pd.qcut(sub.rest, 10, duplicates="drop", labels=False)
    out = np.zeros((2, K))
    w = 0.0
    for b in np.unique(q[~pd.isna(q)]):
        s = sub[q == b]
        t = pd.crosstab(s[group_col], s.response_emotion)
        t = t.reindex(index=lv, columns=EMOTIONS, fill_value=0)
        if (t.sum(axis=1) < 5).any():
            continue
        out += (t.div(t.sum(axis=1), axis=0)).to_numpy() * len(s)
        w += len(s)
    if w == 0:
        return None
    out /= w
    d = out[1] - out[0]
    j = int(np.abs(d).argmax())
    return {
        "matched_response_share": {e: [float(out[0, i]), float(out[1, i])]
                                   for i, e in enumerate(EMOTIONS)},
        "largest_shift_response": EMOTIONS[j],
        "largest_shift": float(d[j]),
        "total_variation_distance": float(np.abs(d).sum() / 2),
    }


# -------------------------------------------------------------------- driver

def run(df: pd.DataFrame, meta: dict, rng, key: str = "intent") -> dict:
    df = add_groups(df, rng)
    if key == "consensus_loo":
        df["intended_emotion"] = loo_consensus(df, rng)
        df = df[df.intended_emotion.notna()].copy()
        df["correct"] = (df.response_emotion.astype(str)
                         == df.intended_emotion.astype(str)).astype(int)
    df["rest"] = rest_scores(df)
    groups = [c for c in df.columns if c.startswith("grp_")]
    res = {**meta, "keyed_response": {"intent": "intended_emotion",
                                      "consensus_loo": "leave-one-rater-out crowd "
                                                       "majority"}[key],
           "style_pc1_ability_correlation": df.attrs.get("style_ability_corr"),
           "matching_variable": "rest score over the other five items",
           "grouping_variables": groups, "items": {}}

    rows = []
    for gcol in groups:
        d = df[df._style_holdout] if gcol.startswith("grp_style") else df
        for e in EMOTIONS:
            sub = d[d.intended_emotion == e]
            r = lr_dif(sub, gcol)
            if r is None:
                continue
            r["item"] = e
            r["mh"] = mantel_haenszel(sub, gcol)
            r["nominal"] = nominal_shift(sub, gcol)
            res["items"].setdefault(gcol, {})[e] = r
            rows.append({
                "group_var": gcol, "item": e, "n_trials": r["n_trials"],
                "n_raters": r["n_raters"],
                "delta_r2": round(r["delta_r2_nagelkerke"], 5),
                "jodoin_gierl": r["jodoin_gierl"], "zumbo_thomas": r["zumbo_thomas"],
                "beta_uniform": round(r["beta_group_uniform"], 4),
                "z_uniform": round(r["z_group_clustered"], 3),
                "p_uniform": r["p_group_clustered"],
                "beta_nonuniform": round(r["beta_interaction_nonuniform"], 4),
                "p_nonuniform": r["p_interaction_clustered"],
                "acc_diff": round(r["raw_accuracy"]["diff"], 4),
                "mh_delta": round(r["mh"]["delta_mh"], 3) if r["mh"] else None,
                "mh_ets": r["mh"]["ets"] if r["mh"] else None,
                "decile_tvd_approx": round(r["nominal"]["total_variation_distance"], 4)
                if r["nominal"] else None,
                "largest_response_shift": r["nominal"]["largest_shift_response"]
                if r["nominal"] else None,
            })

    tab = pd.DataFrame(rows).sort_values(["group_var", "delta_r2"], ascending=[True, False])
    # least invariant class = largest mean effect size across grouping variables
    rank = (tab.groupby("item").delta_r2.mean().sort_values(ascending=False))
    res["least_invariant_items"] = [{"item": k, "mean_delta_r2": float(v)}
                                    for k, v in rank.items()]
    return res, tab


def self_check() -> int:
    """Plant DIF of known sign and size; the machinery must find it and must NOT
    find it where none was planted."""
    rng = np.random.default_rng(0)
    n_r, n_t = 600, 60
    ability = rng.beta(5, 3, n_r)
    grp = rng.integers(0, 2, n_r)
    rows = []
    for r in range(n_r):
        for t in range(n_t):
            e = EMOTIONS[t % K]
            p = ability[r]
            if e == "fear" and grp[r] == 1:
                p = max(p - 0.30, 0.02)          # planted: fear is harder for group 1
            ok = rng.random() < p
            resp = e if ok else EMOTIONS[(t // K + 1) % K]
            rows.append((f"c{t}", f"r{r}", "audio", resp, e, str(r % 9), "male",
                         "S", 3000, t, 50.0))
    df = pd.DataFrame(rows, columns=[
        "clip_id", "rater_id", "presented_modality", "response_emotion",
        "intended_emotion", "actor_id", "actor_sex", "sentence_id",
        "response_time_ms", "question_num", "response_intensity"])
    df["correct"] = (df.response_emotion == df.intended_emotion).astype(int)
    df["grp_planted"] = np.where(pd.Series(grp, index=[f"r{i}" for i in range(n_r)])
                                 .reindex(df.rater_id).to_numpy() == 1, "B", "A")
    df["rest"] = rest_scores(df)

    fear = lr_dif(df[df.intended_emotion == "fear"], "grp_planted")
    assert fear["delta_r2_nagelkerke"] > 0.07, \
        f"planted large DIF not detected: dR2={fear['delta_r2_nagelkerke']:.4f}"
    assert fear["jodoin_gierl"] == "large", f"got {fear['jodoin_gierl']}"
    assert fear["beta_group_uniform"] < 0, "planted DIF has the wrong sign"
    assert fear["p_group_clustered"] < 0.01

    clean = [lr_dif(df[df.intended_emotion == e], "grp_planted")
             for e in EMOTIONS if e != "fear"]
    worst = max(c["delta_r2_nagelkerke"] for c in clean)
    assert worst < 0.035, f"false positive DIF on a clean item: dR2={worst:.4f}"

    mh = mantel_haenszel(df[df.intended_emotion == "fear"], "grp_planted")
    assert mh["ets"].startswith("C"), f"MH missed a large planted effect: {mh}"
    assert abs(mh["delta_mh"]) > 1.5, f"MH delta too small: {mh['delta_mh']:.3f}"

    ns = nominal_shift(df[df.intended_emotion == "fear"], "grp_planted")
    assert ns["total_variation_distance"] > 0.05

    # rest score must exclude the studied item
    d2 = df[df.intended_emotion == "fear"]
    assert d2.rest.corr(d2.correct) < 0.6, "rest score looks contaminated by its own item"

    # leave-one-rater-out consensus must exclude the rater's own vote
    d3 = pd.DataFrame({
        "clip_id": ["c0"] * 4 + ["c1"] * 2,
        "presented_modality": ["audio"] * 4 + ["visual"] * 2,
        "response_emotion": ["fear", "fear", "fear", "sad", "happy", "happy"],
    })
    lc = loo_consensus(d3, np.random.default_rng(0))
    # the lone 'sad' rater must not see their own vote; the 'fear' raters see 2 of 3
    assert list(lc[:4]) == ["fear"] * 4, list(lc[:4])
    assert list(lc[4:]) == ["happy", "happy"], list(lc[4:])
    # a rater whose own vote is in the tied majority must see the tie broken without it
    d5 = pd.DataFrame({"clip_id": ["c0"] * 4, "presented_modality": ["audio"] * 4,
                       "response_emotion": ["fear", "fear", "sad", "sad"]})
    assert list(loo_consensus(d5, np.random.default_rng(0))) == \
        ["sad", "sad", "fear", "fear"], "own vote is leaking into the consensus"
    assert loo_consensus(d5.iloc[[0]], np.random.default_rng(0)).isna().all(), \
        "single-rater clip must yield no leave-one-out consensus"
    print("ok  planted DIF recovered (dR2="
          f"{fear['delta_r2_nagelkerke']:.3f}, MH {mh['ets']}), "
          f"no false positives (max clean dR2={worst:.4f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--modality", default=None, help="audio | visual | audiovisual")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default="intent", choices=["intent", "consensus_loo"],
                    help="score correctness against the actor's intent (default) or "
                         "against the leave-one-rater-out crowd majority")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return self_check()

    rng = np.random.default_rng(a.seed)
    df, meta = load(a.ratings)
    if a.modality:
        df = df[df.presented_modality == a.modality]
        meta = {**meta, "modality_filter": a.modality, "n_ratings": int(len(df))}

    res, tab = run(df, meta, rng, key=a.key)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{a.modality}" if a.modality else ""
    suffix += "-consensusloo" if a.key == "consensus_loo" else ""
    tab.to_csv(OUT / f"dif{suffix}.csv", index=False)
    (OUT / f"dif{suffix}.json").write_text(json.dumps(res, indent=2))

    tag = "  [SIMULATED]" if meta["simulated"] else ""
    print(f"\nsource: {res['source_file']}{tag}   "
          f"modality={a.modality or 'all pooled'}   n={res['n_ratings']:,} trials, "
          f"{res['n_raters']:,} raters")
    print(f"keyed on {res['keyed_response']}; "
          "matched on rest score over the other five items\n")
    show = ["group_var", "item", "n_trials", "delta_r2", "jodoin_gierl",
            "beta_uniform", "z_uniform", "p_nonuniform", "acc_diff",
            "mh_ets", "decile_tvd_approx", "largest_response_shift"]
    print(tab[show].to_string(index=False))
    print("\nleast invariant items (mean dR2 across grouping variables):")
    for r in res["least_invariant_items"]:
        print(f"  {r['item']:9s} {r['mean_delta_r2']:.4f}")
    print(f"\nwrote {OUT / f'dif{suffix}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
