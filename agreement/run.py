"""How well do CREMA-D's crowd raters agree with each other?

Writes agreement/out/agreement.json (machine-readable, every number stamped with
its n) and agreement/out/REPORT.md (the same numbers in prose).

    .venv/bin/python agreement/run.py [--ratings PATH] [--reps 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agreement import (RNG, alpha_fixed_marginals, bootstrap_ci, coincidence_matrix,
                       counts_matrix, fleiss_kappa, krippendorff_alpha)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "agreement" / "out"
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
MODALITIES = ["audio", "visual", "audiovisual"]
CHANCE = 1 / len(EMOTIONS)


def ci(N, reps):
    lo, hi, ok = bootstrap_ci(N, reps=reps)
    return {"alpha": krippendorff_alpha(N), "ci95": [lo, hi],
            "n_units": int((np.asarray(N).sum(axis=1) >= 2).sum()),
            "n_ratings": int(np.asarray(N).sum()), "boot_reps_finite": ok}


def paired_delta(df, mod_a, mod_b, reps, rng=RNG):
    """CI on alpha(mod_a) - alpha(mod_b), resampling *clips* so the two
    conditions stay paired. Every clip was rated in all three modalities, so the
    comparison is within-clip and the difference is not confounded by which
    clips happened to land in which condition."""
    clips = np.sort(df.clip_id.unique())
    idx = {c: i for i, c in enumerate(clips)}
    mats = {}
    for m in (mod_a, mod_b):
        N = np.zeros((len(clips), len(EMOTIONS)), dtype=np.int64)
        sub = df[df.presented_modality == m]
        cm, keys = counts_matrix(sub, ["clip_id"], "response_emotion", EMOTIONS)
        N[[idx[k] for k in keys]] = cm
        mats[m] = N
    obs = krippendorff_alpha(mats[mod_a]) - krippendorff_alpha(mats[mod_b])
    draws = np.empty(reps)
    for i in range(reps):
        take = rng.integers(0, len(clips), len(clips))
        draws[i] = krippendorff_alpha(mats[mod_a][take]) - krippendorff_alpha(mats[mod_b][take])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"delta_alpha": float(obs), "ci95": [float(lo), float(hi)],
            "n_clips": len(clips), "excludes_zero": bool(lo > 0 or hi < 0)}


def per_rater(df):
    """Each rater's agreement with the leave-one-out consensus of the other
    raters on the same clip-modality cell.

    Leave-one-out matters: with 6-12 raters per cell, a rater who is included in
    the consensus they are scored against gets a free vote, and the effect is
    biggest for exactly the cells with fewest raters.

    The null is not 1/6. A rater who clicked with no signal at all, but with
    realistic button-press base rates, would still match the consensus
    sum_c q_c * P(consensus = c) of the time, where q is the pooled response
    distribution. Each rater is tested against their own such expectation
    (it depends on which clips they saw) with an exact binomial test.
    """
    counts = (df.groupby(["clip_id", "presented_modality", "response_emotion"], observed=True)
                .size().unstack("response_emotion", fill_value=0)
                .reindex(columns=EMOTIONS, fill_value=0))
    cell = df.set_index(["clip_id", "presented_modality"])
    loo = counts.loc[pd.MultiIndex.from_frame(df[["clip_id", "presented_modality"]])].to_numpy()
    own = pd.Categorical(df.response_emotion, categories=EMOTIONS).codes
    loo = loo.copy()
    loo[np.arange(len(loo)), own] -= 1                      # drop this rater's own vote
    top = loo.max(axis=1)
    matched = (loo[np.arange(len(loo)), own] == top) & (top > 0)
    tied = (loo == top[:, None]).sum(axis=1) > 1
    # a tie that includes the rater's answer counts as a partial match
    credit = np.where(matched & tied, 1.0 / (loo == top[:, None]).sum(axis=1), matched.astype(float))

    q = df.response_emotion.value_counts(normalize=True).reindex(EMOTIONS).to_numpy()
    cons_share = (loo == top[:, None]) / np.maximum((loo == top[:, None]).sum(axis=1), 1)[:, None]
    expected = (cons_share * q).sum(axis=1)                 # per-rating chance of a match

    out = pd.DataFrame({
        "rater_id": df.rater_id.to_numpy(),
        "modality": df.presented_modality.astype(str).to_numpy(),
        "credit": credit, "expected": expected,
        "rt_ms": df.response_time_ms.to_numpy(),
    })
    from scipy.stats import binomtest
    rows = []
    for rid, g in out.groupby("rater_id", sort=False):
        n, k, e = len(g), g.credit.sum(), g.expected.mean()
        rows.append({
            "rater_id": rid, "n_ratings": n,
            "consensus_agreement": k / n,
            "expected_by_chance": float(e),
            "p_above_chance": binomtest(int(round(k)), n, e, alternative="greater").pvalue,
            "median_rt_ms": float(g.rt_ms.median()),
            **{f"agreement_{m}": float(g.credit[g.modality == m].mean()) if (g.modality == m).any()
               else float("nan") for m in MODALITIES},
        })
    r = pd.DataFrame(rows)
    # Benjamini-Hochberg over 2,443 raters; flagging at raw p would produce ~122
    # false "unreliable" raters by construction.
    p = r.p_above_chance.to_numpy()
    order = np.argsort(p)
    q_bh = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r["q_bh"] = np.nan
    r.loc[r.index[order], "q_bh"] = q_bh
    r["at_chance"] = r.q_bh > 0.05
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(ROOT / "data" / "ratings_long.parquet"))
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    path = Path(args.ratings)
    if not path.exists():
        alt = ROOT / "data" / "SIMULATED_ratings_long.parquet"
        if not alt.exists():
            sys.exit(f"no ratings at {path}; run data_ingest/build_ratings.py")
        print(f"!! {path} missing, falling back to SIMULATED fixture {alt}", file=sys.stderr)
        path = alt
    simulated = path.name.startswith("SIMULATED")
    df = pd.read_parquet(path)
    reps = args.reps

    res = {"source_file": str(path), "simulated": simulated, "bootstrap_reps": reps,
           "n_ratings": len(df), "n_clips": int(df.clip_id.nunique()),
           "n_raters": int(df.rater_id.nunique()), "chance_accuracy": CHANCE}

    # ---------- 1. alpha overall and by modality ----------
    N_all, _ = counts_matrix(df, ["clip_id", "presented_modality"], "response_emotion", EMOTIONS)
    res["alpha_overall"] = ci(N_all, reps)
    res["alpha_overall"]["note"] = ("units are clip-x-modality cells; pools three presentation "
                                    "conditions that differ a lot, so read the by-modality numbers")
    res["alpha_by_modality"] = {}
    mats = {}
    for m in MODALITIES:
        N, _ = counts_matrix(df[df.presented_modality == m], ["clip_id"], "response_emotion", EMOTIONS)
        mats[m] = N
        res["alpha_by_modality"][m] = ci(N, reps)
    res["modality_contrasts"] = {
        "audio_minus_audiovisual": paired_delta(df, "audio", "audiovisual", reps),
        "audio_minus_visual": paired_delta(df, "audio", "visual", reps),
        "visual_minus_audiovisual": paired_delta(df, "visual", "audiovisual", reps),
    }

    # ---------- 2. Fleiss kappa, on the balanced subset where it is defined ----------
    res["fleiss_vs_alpha"] = {
        "note": ("Fleiss' kappa is only defined when every unit has the same number of raters. "
                 "CREMA-D cells hold 6-12, so kappa is computed on the exactly-10 subset and "
                 "alpha is recomputed on that same subset for a like-for-like comparison. "
                 "They differ for two reasons: alpha's expected disagreement carries an "
                 "(n-1) small-sample correction that kappa's does not, and alpha is defined "
                 "over the whole ragged design while kappa has to throw 32% of the cells away."),
        "subset": "clip-modality cells with exactly 10 ratings",
    }
    for m in MODALITIES:
        N = mats[m]
        bal = N[N.sum(axis=1) == 10]
        res["fleiss_vs_alpha"][m] = {
            "n_units_balanced": int(len(bal)),
            "n_units_all": int(len(N)),
            "fleiss_kappa": fleiss_kappa(bal),
            "alpha_same_subset": krippendorff_alpha(bal),
            "alpha_all_units": krippendorff_alpha(N),
        }

    # ---------- 3. alpha per intended-emotion class ----------
    res["alpha_by_class"] = {"note": (
        "Restricting units to one intended emotion also restricts the response marginals, "
        "which shrinks expected disagreement and drags alpha down for reasons that have "
        "nothing to do with the class being hard. alpha_local uses the subset's own "
        "marginals (the usual reported figure, not comparable across classes); "
        "alpha_global_marginals holds expected disagreement at the full-modality value "
        "and IS comparable across classes. Read the second column.")}
    for m in MODALITIES:
        n_c_full = mats[m].sum(axis=0)
        per_class = {}
        for emo in EMOTIONS:
            sub = df[(df.presented_modality == m) & (df.intended_emotion == emo)]
            N, _ = counts_matrix(sub, ["clip_id"], "response_emotion", EMOTIONS)
            c = ci(N, reps)
            c["alpha_local"] = c.pop("alpha")
            c["alpha_global_marginals"] = alpha_fixed_marginals(N, n_c_full)
            c["hit_rate_vs_intended"] = float((sub.response_emotion == emo).mean())
            per_class[emo] = c
        res["alpha_by_class"][m] = per_class

    # ---------- 4. which pairs get conflated ----------
    conf = {"note": ("coincidence[c][k] is how often responses c and k landed on the SAME clip. "
                     "Symmetric, and it never mentions the actor's intent, so it is the honest "
                     "answer to 'what do raters confuse'. Rows normalised to sum to 1.")}
    for m in MODALITIES:
        o = coincidence_matrix(mats[m])
        conf[m] = {
            "coincidence_row_normalised": {
                a: {b: float(o[i, j] / o[i].sum()) for j, b in enumerate(EMOTIONS)}
                for i, a in enumerate(EMOTIONS)},
            "top_confusions": sorted(
                [{"pair": [EMOTIONS[i], EMOTIONS[j]],
                  "share_of_offdiagonal_mass": float(2 * o[i, j] / (o.sum() - np.trace(o)))}
                 for i in range(len(EMOTIONS)) for j in range(i + 1, len(EMOTIONS))],
                key=lambda d: -d["share_of_offdiagonal_mass"])[:5],
        }
        sub = df[df.presented_modality == m]
        ct = pd.crosstab(sub.intended_emotion, sub.response_emotion, normalize="index")
        conf[m]["response_given_intended"] = ct.reindex(index=EMOTIONS, columns=EMOTIONS).round(4).to_dict()
    res["confusion"] = conf

    # ---------- 5. does portrayed intensity help? ----------
    # Levels exist only for sentence IEO and only for the 5 non-neutral emotions:
    # 91 actors x 5 emotions x 3 levels = 455 clips per level, balanced by actor
    # and emotion. Everything else is level "unspecified". So the contrast is
    # within-sentence and within-actor by construction.
    ieo = df[(df.sentence_id == "IEO") & (df.intended_intensity != "unspecified")]
    res["intensity"] = {
        "design_note": ("intensity levels exist only for sentence IEO and only for the five "
                        "non-neutral emotions -- 91 actors x 5 emotions x 3 levels = 455 clips "
                        "per level, balanced on actor and emotion. Neutral has no level. "
                        "This is a within-sentence contrast, not the whole corpus."),
        "clips_per_level": {k: int(v) for k, v in
                            ieo.groupby("intended_intensity", observed=True).clip_id.nunique().items()},
    }
    for m in MODALITIES:
        per_level = {}
        for lvl in ["low", "medium", "high"]:
            sub = ieo[(ieo.presented_modality == m) & (ieo.intended_intensity == lvl)]
            N, _ = counts_matrix(sub, ["clip_id"], "response_emotion", EMOTIONS)
            c = ci(N, reps)
            c["hit_rate_vs_intended"] = float((sub.response_emotion == sub.intended_emotion).mean())
            per_level[lvl] = c
        res["intensity"][m] = per_level

    # high-vs-low is paired on (actor, emotion): the same actor performed the
    # same emotion at all three levels, so resampling those 455 cells keeps the
    # contrast within-actor and within-emotion.
    def intensity_delta(m, hi="high", lo="low", rng=RNG):
        sub = ieo[ieo.presented_modality == m]
        cells = sub[["actor_id", "intended_emotion"]].drop_duplicates().reset_index(drop=True)
        key = pd.MultiIndex.from_frame(cells)
        mats_lvl = {}
        for lvl in (hi, lo):
            s2 = sub[sub.intended_intensity == lvl]
            N = np.zeros((len(cells), len(EMOTIONS)), dtype=np.int64)
            cm, idxk = counts_matrix(s2, ["actor_id", "intended_emotion"], "response_emotion", EMOTIONS)
            N[key.get_indexer(idxk)] = cm
            mats_lvl[lvl] = N
        obs = krippendorff_alpha(mats_lvl[hi]) - krippendorff_alpha(mats_lvl[lo])
        draws = np.empty(reps)
        for i in range(reps):
            t = rng.integers(0, len(cells), len(cells))
            draws[i] = krippendorff_alpha(mats_lvl[hi][t]) - krippendorff_alpha(mats_lvl[lo][t])
        lo_ci, hi_ci = np.nanpercentile(draws, [2.5, 97.5])
        return {"delta_alpha_high_minus_low": float(obs), "ci95": [float(lo_ci), float(hi_ci)],
                "n_actor_emotion_cells": int(len(cells)), "excludes_zero": bool(lo_ci > 0 or hi_ci < 0)}

    res["intensity"]["high_minus_low_paired"] = {m: intensity_delta(m) for m in MODALITIES}

    # framing test: is acted+exaggerated actually the optimistic end?
    a = res["intensity"]["audio"]
    res["framing_test_acted_is_the_optimistic_case"] = {
        "claim": ("CREMA-D is acted, exaggerated emotion, so it should be the optimistic case; "
                  "natural speech would be worse."),
        "test": "alpha on deliberately HIGH-intensity portrayals vs deliberately LOW ones, audio-only, IEO clips",
        "alpha_high": a["high"]["alpha"], "ci95_high": a["high"]["ci95"],
        "alpha_low": a["low"]["alpha"], "ci95_low": a["low"]["ci95"],
        "n_clips_per_level": res["intensity"]["clips_per_level"],
        "paired_delta": res["intensity"]["high_minus_low_paired"]["audio"],
        "reference": ("Krippendorff's own thresholds: alpha >= 0.800 to draw firm conclusions, "
                      ">= 0.667 for tentative ones. The deliberately-exaggerated audio-only "
                      "ceiling sits well under both."),
    }

    # ---------- 6. per-rater reliability ----------
    r = per_rater(df)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "per_rater.csv", index=False)
    qs = r.consensus_agreement.quantile([0, .05, .25, .5, .75, .95, 1]).round(4)
    res["per_rater"] = {
        "note": ("agreement with the leave-one-out consensus of the other raters on the same "
                 "clip-modality cell; ties split proportionally. 'at chance' = fails a one-sided "
                 "exact binomial test against that rater's own expected match rate under "
                 "marginal-matched random responding, at BH q<0.05 across all raters."),
        "n_raters": int(len(r)),
        "mean": float(r.consensus_agreement.mean()),
        "sd": float(r.consensus_agreement.std()),
        "quantiles": {str(k): float(v) for k, v in qs.items()},
        "mean_expected_by_chance": float(r.expected_by_chance.mean()),
        "n_at_chance": int(r.at_chance.sum()),
        "pct_at_chance": float(r.at_chance.mean() * 100),
        "n_below_chance_point_estimate": int((r.consensus_agreement < r.expected_by_chance).sum()),
        "by_modality_mean": {m: float(r[f"agreement_{m}"].mean()) for m in MODALITIES},
        "median_rt_ms_at_chance_vs_rest": [
            float(r.loc[r.at_chance, "median_rt_ms"].median()),
            float(r.loc[~r.at_chance, "median_rt_ms"].median())],
    }
    # what happens to alpha if the at-chance raters are removed?
    keep = set(r.loc[~r.at_chance, "rater_id"])
    clean = df[df.rater_id.isin(keep)]
    res["per_rater"]["alpha_after_dropping_at_chance_raters"] = {}
    for m in MODALITIES:
        N, _ = counts_matrix(clean[clean.presented_modality == m], ["clip_id"], "response_emotion", EMOTIONS)
        res["per_rater"]["alpha_after_dropping_at_chance_raters"][m] = {
            "alpha": krippendorff_alpha(N), "n_ratings": int(N.sum()),
            "alpha_before": res["alpha_by_modality"][m]["alpha"]}

    (OUT / "agreement.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in
                      ["alpha_overall", "alpha_by_modality", "modality_contrasts"]},
                     indent=2, default=float))
    print(f"\nwrote {OUT/'agreement.json'} and {OUT/'per_rater.csv'}")
    return res


if __name__ == "__main__":
    main()
