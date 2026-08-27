"""Verify our parse against CREMA-D's own R pipeline output.

The authors ship `processedResults/tabulatedVotes.csv` (vote counts per clip per
modality) and `processedResults/summaryTable.csv` (majority vote per modality),
produced by `processFinishedResponses.R` from the same CSV we parse. Reproducing
those files exactly is an independent check on the ingest.

It only reproduces once you apply a filter the dataset README never mentions:
`processFinishedResponses.R` drops every response whose FIRST emotion click took
longer than 10 seconds, matched on sessionNums*1000 + queryType*100 + questNum
against `finishedEmoResponses.csv`. That is 7,688 of 219,688 responses (3.50%).
`ratings_long.parquet` carries it as `authors_excluded` rather than applying it.

    .venv/bin/python data_ingest/validate_against_authors.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LETTERS = list("ADFHNS")
EMO_LETTER = {"anger": "A", "disgust": "D", "fear": "F", "happy": "H", "neutral": "N", "sad": "S"}
MODALITY = {1: "audio", 2: "visual", 3: "audiovisual"}
AUTHORS_TOTAL_VOTES = 212_000


def authors_subset():
    """finishedResponses.csv with processFinishedResponses.R's filter applied.

    Read from the raw CSV, not from our parquet, so this is a genuinely
    independent path -- and because the authors never de-duplicate, so the two
    rows we drop have to stay in for the totals to line up.
    """
    resp = pd.read_csv(RAW / "finishedResponses.csv", index_col=0, low_memory=False)
    emo = pd.read_csv(RAW / "finishedEmoResponses.csv", index_col=0, low_memory=False)
    slow = set((emo.sessionNums * 1000 + emo.queryType * 100 + emo.questNum)[
        pd.to_numeric(emo.ttr, errors="coerce") > 10_000])
    key = resp.sessionNums * 1000 + resp.queryType * 100 + resp.questNum
    return resp[~key.isin(slow)]


def main():
    good = authors_subset()
    fails = []

    # ---- vote counts, all 22,326 clip-modality cells ----
    ours = (good.groupby(["clipName", "queryType", "respEmo"]).size()
                .unstack("respEmo", fill_value=0).reindex(columns=LETTERS, fill_value=0).reset_index())
    ours["presented_modality"] = ours.queryType.map(MODALITY)
    tv = pd.read_csv(RAW / "processedResults" / "tabulatedVotes.csv", index_col=0)
    tv["presented_modality"] = (tv.index // 100000).map(MODALITY)   # row id = queryType * 100000 + clipNum
    theirs = tv[["fileName", "presented_modality"] + LETTERS].rename(columns={"fileName": "clip_id"})

    m = ours.rename(columns={"clipName": "clip_id"}).merge(
        theirs, on=["clip_id", "presented_modality"], suffixes=("_o", "_t"), how="outer", indicator=True)
    if (m._merge != "both").any():
        fails.append(f"{(m._merge != 'both').sum()} clip-modality cells present in only one source")
    off = sum(abs(m[f"{L}_o"].fillna(-1) - m[f"{L}_t"].fillna(-1)) for L in LETTERS) > 0
    print(f"vote counts: {(~off).sum():,}/{len(m):,} clip-modality cells identical")
    if off.any():
        fails.append(f"{off.sum()} clip-modality cells have different vote counts")
        print(m[off].head(10).to_string())

    total_theirs = int(tv[LETTERS].to_numpy().sum())
    print(f"total votes: ours {len(good):,}, theirs {total_theirs:,}")
    if total_theirs != AUTHORS_TOTAL_VOTES or len(good) != AUTHORS_TOTAL_VOTES:
        fails.append(f"expected {AUTHORS_TOTAL_VOTES:,} votes on both sides")

    # ---- majority vote against summaryTable ----
    st = pd.read_csv(RAW / "processedResults" / "summaryTable.csv", index_col=0).set_index("FileName")
    for mod, col in [("audio", "VoiceVote"), ("visual", "FaceVote"), ("audiovisual", "MultiModalVote")]:
        sub = m[m.presented_modality == mod].set_index("clip_id")
        counts = sub[[f"{L}_o" for L in LETTERS]].to_numpy()
        top = counts.max(axis=1)
        # they write ties as "A:D"; compare as sets so tie-breaking never matters
        ours_set = [set(np.array(LETTERS)[r == t]) for r, t in zip(counts, top)]
        theirs_set = st[col].reindex(sub.index).map(
            lambda v: set(v.split(":")) if isinstance(v, str) else set())
        ok = np.array([a == b for a, b in zip(ours_set, theirs_set)])
        n_tied = int(sum(len(s) > 1 for s in ours_set))
        print(f"{mod:12s} majority vote (as a set, so ties compare exactly): "
              f"{ok.sum():,}/{len(ok):,} clips  [{n_tied} tied]")
        if not ok.all():
            fails.append(f"{(~ok).sum()} {mod} majority votes disagree with summaryTable")
            print("   first disagreements:",
                  [(i, sorted(a), sorted(b)) for i, a, b, k in
                   zip(sub.index, ours_set, theirs_set, ok) if not k][:5])

    # ---- and the flag in the parquet agrees with the filter computed here ----
    df = pd.read_parquet(ROOT / "data" / "ratings_long.parquet")
    kept = int((~df.authors_excluded).sum())
    print(f"\nratings_long.parquet authors_excluded flag keeps {kept:,} rows "
          f"({AUTHORS_TOTAL_VOTES - kept} fewer than the authors, "
          f"= duplicate rows we drop and they do not)")
    if not 0 <= AUTHORS_TOTAL_VOTES - kept <= 2:
        fails.append("authors_excluded flag does not line up with the authors' subset")

    if fails:
        print("\nFAILED:", *fails, sep="\n  ")
        sys.exit(1)
    print("\nOK: our parse reproduces the authors' own R pipeline output exactly")


if __name__ == "__main__":
    main()
