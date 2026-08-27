"""Shared loading + schema for ceiling/ and invariance/.

Owned by agent C (see CONTRACT.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "ratings_long.parquet"
SIM = ROOT / "data" / "SIMULATED_ratings_long.parquet"

EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
E_IDX = {e: i for i, e in enumerate(EMOTIONS)}
K = len(EMOTIONS)

REQUIRED = [
    "clip_id", "rater_id", "presented_modality", "response_emotion",
    "intended_emotion", "actor_id", "actor_sex", "sentence_id",
]


def resolve(path: str | None) -> tuple[Path, bool]:
    """Return (path, is_simulated). Real data wins; fixture is the loud fallback."""
    if path:
        p = Path(path)
        if not p.exists():
            sys.exit(f"no such ratings file: {p}")
        return p, "SIMULATED" in p.name
    if REAL.exists():
        return REAL, False
    if SIM.exists():
        print(
            "\n*** WARNING: real data/ratings_long.parquet not found. "
            "Using SIMULATED fixture. Every number below is FAKE. ***\n",
            file=sys.stderr,
        )
        return SIM, True
    sys.exit(
        "no ratings file. Run: .venv/bin/python ceiling/simulate.py  "
        "(or wait for data/ratings_long.parquet)"
    )


def load(path: str | None = None) -> tuple[pd.DataFrame, dict]:
    p, sim = resolve(path)
    df = pd.read_parquet(p)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{p} violates CONTRACT.md, missing columns: {missing}")
    for c in ("response_emotion", "intended_emotion"):
        bad = set(df[c].dropna().unique()) - set(EMOTIONS)
        if bad:
            sys.exit(f"{p}: {c} has non-contract values {sorted(bad)}; expected {EMOTIONS}")
    df = df.dropna(subset=["clip_id", "rater_id", "response_emotion"])
    meta = {
        "source_file": str(p.relative_to(ROOT)),
        "simulated": bool(sim),
        "n_ratings": int(len(df)),
        "n_clips": int(df.clip_id.nunique()),
        "n_raters": int(df.rater_id.nunique()),
        "n_actors": int(df.actor_id.nunique()),
        "modalities": sorted(df.presented_modality.dropna().unique().tolist()),
    }
    return df, meta


def count_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """clips x K matrix of response counts. Returns (counts, clip_ids)."""
    clips = np.sort(df.clip_id.unique())
    ci = pd.Series(np.arange(len(clips)), index=clips)
    counts = np.zeros((len(clips), K), dtype=np.int64)
    np.add.at(
        counts,
        (ci.loc[df.clip_id].to_numpy(), df.response_emotion.map(E_IDX).to_numpy()),
        1,
    )
    return counts, clips


def majority_label(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Argmax with ties broken uniformly at random. Returns class indices."""
    mx = counts.max(axis=1, keepdims=True)
    tied = counts == mx
    # random tiebreak: add uniform noise only among tied cells
    u = rng.random(counts.shape) * tied
    return u.argmax(axis=1)


def n_tied(counts: np.ndarray) -> int:
    return int(((counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())
