"""Turn CREMA-D's raw per-rater response log into tidy parquet.

Inputs  (data/raw/, fetched by data_ingest/fetch.sh):
    finishedResponses.csv   one row per rater x clip x modality response
    VideoDemographics.csv   actor demographics
    SentenceFilenames.csv   canonical clip list

Outputs:
    data/ratings_long.parquet   one row per response
    data/clips.parquet          one row per clip
    data/ingest_report.json     counts, drops, reconciliation vs the paper

Run: .venv/bin/python data_ingest/build_ratings.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

# CREMA-D README: queryType 1 = voice only, 2 = face only, 3 = audio-visual.
# Names on the right are CONTRACT.md's vocabulary, not CREMA-D's.
MODALITY = {1: "audio", 2: "visual", 3: "audiovisual"}
# CREMA-D's single-letter response/display codes -> CONTRACT.md vocabulary.
EMO = {"A": "anger", "D": "disgust", "F": "fear", "H": "happy", "N": "neutral", "S": "sad"}
EMOTIONS = list(EMO.values())
# Third filename field -> the same vocabulary.
EMO_FROM_FILENAME = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
                     "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
# Fourth filename field / dispLevel letter.
INTENSITY = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}

# Claims made by the dataset's own README / the CREMA-D paper (Cao et al. 2014,
# IEEE Trans. Affective Computing 5(4):377-390). Every one of these is checked.
PAPER = {
    "n_clips": 7442,
    "n_actors": 91,
    "n_actors_male": 48,
    "n_actors_female": 43,
    "n_raters": 2443,
    "clips_per_rater": 90,
    "clips_per_rater_per_modality": 30,
    "n_sentences": 12,
    "min_actor_age": 20,
    "max_actor_age": 74,
}


def build():
    drops = []  # every dropped row is recorded here and lands in data/README.md

    # dtype=str throughout: respLevel has 3 malformed cells, and letting pandas
    # infer would silently coerce them to NaN without telling us which.
    resp = pd.read_csv(
        RAW / "finishedResponses.csv", index_col=0, dtype=str,
        keep_default_na=False, na_values=["NA"],
    )
    n_raw = len(resp)

    # The authors' published aggregation (processedResults/*.csv) is NOT computed
    # over all responses. processFinishedResponses.R silently drops every response
    # whose FIRST emotion click took more than 10 s, matched on
    # sessionNums*1000 + queryType*100 + questNum. That is 7,688 responses (3.5%),
    # and it is not mentioned in the dataset README. Flagging it here rather than
    # applying it: reproducing their subset is a one-line filter, and it is worth
    # knowing that the widely-used CREMA-D labels exclude the slowest judgements.
    emo = pd.read_csv(RAW / "finishedEmoResponses.csv", index_col=0, low_memory=False)
    emo_ttr = pd.to_numeric(emo.ttr, errors="coerce")
    slow_keys = set((emo.sessionNums * 1000 + emo.queryType * 100 + emo.questNum)[emo_ttr > 10_000])
    resp_key = (resp.sessionNums.astype(int) * 1000 + resp.queryType.astype(int) * 100
                + resp.questNum.astype(int))
    authors_excluded = resp_key.isin(slow_keys)

    demo = pd.read_csv(RAW / "VideoDemographics.csv")
    sentences = pd.read_csv(RAW / "SentenceFilenames.csv")

    # --- clip identity comes from the filename, not from dispEmo/dispVal ---
    # dispEmo/dispLevel agree with the filename on 100% of rows (checked below),
    # but dispVal is NA on 540 rows while the filename never is.
    parts = resp.clipName.str.split("_", expand=True)
    parts.columns = ["actor_id", "sentence_id", "emo3", "lvl2"]
    intended_emotion = parts.emo3.map(EMO_FROM_FILENAME)
    intended_intensity = parts.lvl2.map(INTENSITY)

    checks = {
        "dispEmo_matches_filename": int((intended_emotion != resp.dispEmo.map(EMO)).sum()),
        "dispLevel_matches_filename": int(
            (parts.lvl2.map({"LO": "L", "MD": "M", "HI": "H", "XX": "X"}) != resp.dispLevel).sum()
        ),
        "ans_field_matches_respEmo": int((resp.ans.str.split("_").str[0] != resp.respEmo).sum()),
        "dispVal_missing": int(resp.dispVal.isna().sum()),
        "subType_all_4": bool((resp.subType == "4").all()),
        "authors_excluded_rows": int(authors_excluded.sum()),
        "authors_excluded_pct": round(float(authors_excluded.mean()) * 100, 3),
        "authors_kept_rows": int((~authors_excluded).sum()),
    }

    df = pd.DataFrame({
        "clip_id": resp.clipName,
        "rater_id": resp.localid,
        "presented_modality": resp.queryType.astype(int).map(MODALITY),
        "response_emotion": resp.respEmo.map(EMO),
        "response_intensity": pd.to_numeric(resp.respLevel, errors="coerce"),
        "intended_emotion": intended_emotion,
        "intended_intensity": intended_intensity,
        "actor_id": parts.actor_id,
        "sentence_id": parts.sentence_id,
        "response_time_ms": resp.ttr.astype(int),
        "num_tries": resp.numTries.astype(int),
        "session_num": resp.sessionNums.astype(int),
        "question_num": resp.questNum.astype(int),
        "log_pos": resp.pos.astype(int),
        "authors_excluded": authors_excluded.to_numpy(),
    })

    # 3 rows carry a letter where the 0-100 intensity slider value should be
    # (the `ans` field is e.g. "H" instead of "H_54"). The emotion is intact, so
    # keep the row and null the intensity rather than throw the rating away.
    n_bad_intensity = int(df.response_intensity.isna().sum())
    drops.append({
        "rule": "malformed respLevel kept with NaN intensity",
        "rows_dropped": 0,
        "rows_affected": n_bad_intensity,
        "detail": "respLevel held an emotion letter, not a 1-100 value; emotion kept, intensity nulled",
    })

    # 2 (rater, clip, modality) cells have two responses. One is a byte-identical
    # replay of the same log line; the other is a rater who changed their answer
    # (numTries=1 on the second). Keep the last response by log position, which is
    # the "final emotional response" the file is documented to contain.
    dup_mask = df.duplicated(subset=["rater_id", "clip_id", "presented_modality"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    df = df.sort_values("log_pos").drop_duplicates(
        subset=["rater_id", "clip_id", "presented_modality"], keep="last"
    )
    drops.append({
        "rule": "duplicate (rater, clip, modality) response",
        "rows_dropped": n_dup_rows - n_dup_rows // 2,
        "rows_affected": n_dup_rows,
        "detail": "kept the last response by log position (the final answer)",
    })

    df = df.merge(
        demo.assign(actor_id=demo.ActorID.astype(str), Sex=demo.Sex.str.lower()).rename(columns={
            "Age": "actor_age", "Sex": "actor_sex",
            "Race": "actor_race", "Ethnicity": "actor_ethnicity",
        })[["actor_id", "actor_age", "actor_sex", "actor_race", "actor_ethnicity"]],
        on="actor_id", how="left", validate="many_to_one",
    )
    assert df.actor_sex.notna().all(), "actor missing from VideoDemographics.csv"

    df = df[[
        "clip_id", "rater_id", "presented_modality", "response_emotion", "response_intensity",
        "intended_emotion", "intended_intensity", "actor_id", "actor_sex", "actor_age",
        "actor_race", "actor_ethnicity", "sentence_id",
        "response_time_ms", "num_tries", "session_num", "question_num", "log_pos",
        "authors_excluded",
    ]].sort_values(["clip_id", "presented_modality", "rater_id"]).reset_index(drop=True)

    for col in ["presented_modality", "response_emotion", "intended_emotion", "intended_intensity",
                "actor_sex", "actor_race", "actor_ethnicity", "sentence_id"]:
        df[col] = df[col].astype("category")

    # --- clip grain: one row per clip, with per-modality vote counts ---
    clips = (
        df.groupby("clip_id", observed=True)
        .agg(actor_id=("actor_id", "first"), actor_sex=("actor_sex", "first"),
             actor_age=("actor_age", "first"), actor_race=("actor_race", "first"),
             actor_ethnicity=("actor_ethnicity", "first"), sentence_id=("sentence_id", "first"),
             intended_emotion=("intended_emotion", "first"),
             intended_intensity=("intended_intensity", "first"),
             n_ratings=("rater_id", "size"))
        .reset_index()
    )
    per_mod = df.pivot_table(index="clip_id", columns="presented_modality", values="rater_id",
                             aggfunc="size", observed=True)
    per_mod.columns = [f"n_ratings_{c}" for c in per_mod.columns]
    clips = clips.merge(per_mod.reset_index(), on="clip_id")

    # Modal response and its share, per modality. Ties broken by the fixed
    # EMOTIONS order so the column is deterministic; tie flag kept alongside.
    for mod in MODALITY.values():
        sub = df[df.presented_modality == mod]
        counts = sub.pivot_table(index="clip_id", columns="response_emotion", values="rater_id",
                                 aggfunc="size", observed=False).reindex(columns=EMOTIONS).fillna(0)
        top = counts.max(axis=1)
        clips = clips.merge(
            pd.DataFrame({
                f"consensus_{mod}": counts.idxmax(axis=1),
                f"agreement_{mod}": top / counts.sum(axis=1),
                f"consensus_tied_{mod}": counts.eq(top, axis=0).sum(axis=1) > 1,
            }).reset_index(), on="clip_id", how="left")

    clips["wav_file"] = clips.clip_id + ".wav"

    # ---------------- reconciliation against the paper ----------------
    obs = {
        "n_clips": int(df.clip_id.nunique()),
        "n_actors": int(df.actor_id.nunique()),
        "n_actors_male": int((demo.Sex == "Male").sum()),
        "n_actors_female": int((demo.Sex == "Female").sum()),
        "n_raters": int(df.rater_id.nunique()),
        "n_sentences": int(df.sentence_id.nunique()),
        "min_actor_age": int(demo.Age.min()),
        "max_actor_age": int(demo.Age.max()),
    }
    per_rater = df.groupby("rater_id", observed=True).size()
    per_rater_mod = df.groupby(["rater_id", "presented_modality"], observed=True).size()

    expected_responses = PAPER["n_raters"] * PAPER["clips_per_rater"]
    recon = {
        "rows_in_source_csv": n_raw,
        "rows_in_ratings_long": len(df),
        "expected_responses_if_every_rater_did_90": expected_responses,
        "shortfall_vs_expected": expected_responses - len(df),
        "raters_with_fewer_than_90": int((per_rater < PAPER["clips_per_rater"]).sum()),
        "raters_with_exactly_90": int((per_rater == PAPER["clips_per_rater"]).sum()),
        "min_responses_by_a_rater": int(per_rater.min()),
        "rater_modality_cells_not_30": int((per_rater_mod != 30).sum()),
        "clip_modality_cells": int(len(df.groupby(["clip_id", "presented_modality"], observed=True))),
        "clip_modality_cells_expected": PAPER["n_clips"] * 3,
        "readme_claim_95pct_clips_over_7_ratings": {
            "claim": ">7 ratings for 95% of clips",
            "observed_pct_clip_modality_cells_over_7": round(
                float((df.groupby(["clip_id", "presented_modality"], observed=True).size() > 7).mean()) * 100, 2),
            "observed_pct_clips_over_7_total_ratings": round(
                float((df.groupby("clip_id", observed=True).size() > 7).mean()) * 100, 2),
        },
        "sentence_filenames_rows": len(sentences),
        "sentence_filenames_match_rated_clips": bool(
            set(sentences.Filename) == set(df.clip_id.unique())),
    }
    mismatches = {k: {"paper": PAPER[k], "observed": v} for k, v in obs.items() if PAPER[k] != v}

    report = {
        "source": "github.com/CheyneyComputerScience/CREMA-D (master)",
        "paper": "Cao et al. 2014, IEEE Trans. Affective Computing 5(4):377-390",
        "paper_claims": PAPER,
        "observed": obs,
        "claim_mismatches": mismatches,
        "reconciliation": recon,
        "source_consistency_checks": checks,
        "drops": drops,
        "ratings_per_clip_modality": {
            str(k): int(v) for k, v in
            df.groupby(["clip_id", "presented_modality"], observed=True).size().value_counts().sort_index().items()
        },
        "emotion_codes": EMO,
        "intensity_codes": INTENSITY,
    }

    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "ratings_long.parquet", index=False)
    clips.to_parquet(OUT / "clips.parquet", index=False)
    (OUT / "ingest_report.json").write_text(json.dumps(report, indent=2))
    return df, clips, report


def selfcheck(df, clips):
    """Parquet round-trips, and per-clip rater counts match the source CSV."""
    back = pd.read_parquet(OUT / "ratings_long.parquet")
    assert len(back) == len(df), (len(back), len(df))
    assert list(back.columns) == list(df.columns)
    pd.testing.assert_frame_equal(back.reset_index(drop=True), df.reset_index(drop=True))

    # Counts recomputed straight off the raw CSV, not off anything derived above.
    raw = pd.read_csv(RAW / "finishedResponses.csv", index_col=0, dtype=str,
                      keep_default_na=False, na_values=["NA"])
    raw = raw.drop_duplicates(subset=["localid", "clipName", "queryType"], keep="last")
    src = raw.groupby("clipName").size().sort_index()
    got = back.groupby("clip_id", observed=True).size().sort_index()
    assert src.equals(got), f"per-clip counts differ on {(src != got).sum()} clips"

    src_mod = raw.assign(m=raw.queryType.astype(int).map(MODALITY)).groupby(["clipName", "m"]).size()
    got_mod = back.groupby(["clip_id", "presented_modality"], observed=True).size()
    assert src_mod.values.tolist() == got_mod.values.tolist(), "per-clip-modality counts differ"

    cb = pd.read_parquet(OUT / "clips.parquet")
    assert len(cb) == PAPER["n_clips"] == back.clip_id.nunique()
    assert cb.n_ratings.sum() == len(back)
    assert set(back.presented_modality.unique()) == {"audio", "visual", "audiovisual"}
    assert back.notna().all().all() or set(back.columns[back.isna().any()]) == {"response_intensity"}, \
        "CONTRACT.md: only response_intensity may be null"
    assert back.response_emotion.dropna().isin(EMOTIONS).all()
    assert back.intended_emotion.isin(EMOTIONS).all()
    iv = back.response_intensity.dropna()
    assert iv.between(1, 100).all(), (iv.min(), iv.max())
    # The authors' own tabulation totals 212,000 votes. We land one short because
    # they do not de-duplicate and one of the two duplicate rows we drop survived
    # their 10 s filter. validate_against_authors.py reproduces their 212,000
    # exactly off the raw CSV and checks all 22,326 cells.
    assert int((~back.authors_excluded).sum()) == 211_999, int((~back.authors_excluded).sum())
    print("selfcheck: OK")


if __name__ == "__main__":
    df, clips, report = build()
    print(f"ratings_long.parquet  {len(df):,} rows x {len(df.columns)} cols")
    print(f"clips.parquet         {len(clips):,} rows x {len(clips.columns)} cols")
    if report["claim_mismatches"]:
        print("\n!! MISMATCH vs the dataset's own published counts:", file=sys.stderr)
        for k, v in report["claim_mismatches"].items():
            print(f"   {k}: paper says {v['paper']}, we observe {v['observed']}", file=sys.stderr)
    else:
        print("all paper-claimed counts reproduce exactly")
    r = report["reconciliation"]
    if r["shortfall_vs_expected"]:
        print(f"\n!! {r['shortfall_vs_expected']} responses short of "
              f"{r['expected_responses_if_every_rater_did_90']} "
              f"(= {PAPER['n_raters']} raters x 90 clips). "
              f"{r['raters_with_fewer_than_90']} raters have partial sessions "
              f"(min {r['min_responses_by_a_rater']} responses). "
              f"Present in the source CSV, not introduced here.", file=sys.stderr)
    selfcheck(df, clips)
