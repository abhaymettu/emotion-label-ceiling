"""Generate data/SIMULATED_ratings_long.parquet — a fixture, not data.

Exists so ceiling/ and invariance/ can be built and tested before the real
CREMA-D ingest lands. Shape and column contract match CONTRACT.md. Every
number it produces is fake and every consumer stamps simulated=true.

The generative model deliberately contains (a) clip-level ambiguity and
(b) rater-level response bias, so the DIF machinery has a known ground truth
to recover. `--check` asserts that it does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EMOTIONS, K, ROOT  # noqa: E402

# fake CREMA-D-shaped constants
N_ACTORS, N_SENTENCES = 91, 12
SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI",
             "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]
MODALITIES = ["audio", "visual", "audiovisual"]
# how legible each intended emotion is, i.e. mass on the intended class before
# rater bias. Loosely ordered the way acted-emotion studies usually find it.
LEGIBILITY = {"anger": 0.80, "happy": 0.62, "sad": 0.60,
              "neutral": 0.72, "fear": 0.45, "disgust": 0.40}
# two latent rater styles with different confusion tendencies -> real DIF
STYLE_TILT = {
    0: {"fear": 1.0, "sad": 1.6, "disgust": 0.6},   # hears distress as sadness
    1: {"fear": 1.7, "sad": 0.7, "disgust": 1.4},   # hears distress as fear
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=7442)
    ap.add_argument("--raters", type=int, default=2443)
    ap.add_argument("--ratings-per-clip", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="run the self-check and exit")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    actors = rng.integers(1001, 1001 + N_ACTORS, size=a.clips)
    sents = rng.integers(0, N_SENTENCES, size=a.clips)
    intended = rng.choice(EMOTIONS, size=a.clips)
    clip_ids = [f"{actors[i]}_{SENTENCES[sents[i]]}_{intended[i][:3].upper()}_{i:05d}"
                for i in range(a.clips)]

    actor_sex = {int(x): ("male" if rng.random() < 0.5 else "female")
                 for x in np.unique(actors)}
    actor_age = {int(x): int(rng.integers(20, 75)) for x in np.unique(actors)}
    actor_race = {int(x): str(rng.choice(["Caucasian", "African American", "Asian",
                                          "Unknown"], p=[.5, .25, .15, .10]))
                  for x in np.unique(actors)}

    rater_ids = np.array([f"W{i:05d}" for i in range(a.raters)])
    rater_style = rng.integers(0, 2, size=a.raters)
    # rater "attentiveness": scales how much mass stays on the legible answer
    rater_att = np.clip(rng.beta(6, 2, size=a.raters), 0.15, 1.0)
    rater_modality = rng.choice(MODALITIES, size=a.raters, p=[0.42, 0.29, 0.29])

    n = a.clips * a.ratings_per_clip
    row_clip = np.repeat(np.arange(a.clips), a.ratings_per_clip)
    # crowd work is Zipf-ish: a few workers do a lot
    w = 1.0 / (1.0 + np.arange(a.raters)) ** 0.55
    row_rater = rng.choice(a.raters, size=n, p=w / w.sum())

    ei = {e: i for i, e in enumerate(EMOTIONS)}
    base = np.zeros((a.clips, K))
    for i, e in enumerate(intended):
        base[i] = (1.0 - LEGIBILITY[e]) / (K - 1)
        base[i, ei[e]] = LEGIBILITY[e]
    # clip-level difficulty jitter: some clips are far more ambiguous than others
    base = base ** rng.uniform(0.35, 1.9, size=(a.clips, 1))
    base /= base.sum(1, keepdims=True)

    tilt = np.ones((2, K))
    for s, d in STYLE_TILT.items():
        for e, m in d.items():
            tilt[s, ei[e]] = m

    p = base[row_clip] * tilt[rater_style[row_rater]]
    att = rater_att[row_rater][:, None]
    p = p ** att                       # low attentiveness -> flatter -> noisier
    p /= p.sum(1, keepdims=True)
    # vectorised categorical sampling
    resp = (p.cumsum(1) > rng.random(n)[:, None]).argmax(1)

    df = pd.DataFrame({
        "clip_id": pd.Categorical([clip_ids[i] for i in row_clip]),
        "rater_id": pd.Categorical(rater_ids[row_rater]),
        "presented_modality": pd.Categorical(rater_modality[row_rater]),
        "response_emotion": pd.Categorical([EMOTIONS[i] for i in resp],
                                           categories=EMOTIONS),
        "response_intensity": rng.choice(["low", "medium", "high"], size=n),
        "intended_emotion": pd.Categorical(intended[row_clip], categories=EMOTIONS),
        "actor_id": pd.Categorical([str(actors[i]) for i in row_clip]),
        "actor_sex": pd.Categorical([actor_sex[int(actors[i])] for i in row_clip]),
        "actor_age": [actor_age[int(actors[i])] for i in row_clip],
        "actor_race": pd.Categorical([actor_race[int(actors[i])] for i in row_clip]),
        "sentence_id": pd.Categorical([SENTENCES[sents[i]] for i in row_clip]),
    })

    if a.check:
        return check(df, rater_style, rater_ids)

    out = ROOT / "data" / "SIMULATED_ratings_long.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    hit = (df.response_emotion == df.intended_emotion).mean()
    print(f"wrote {out}  rows={len(df):,} clips={df.clip_id.nunique():,} "
          f"raters={df.rater_id.nunique():,}  P(response==intended)={hit:.3f}")
    print("SIMULATED. No number derived from this file may appear unlabelled.")
    return 0


def check(df, rater_style, rater_ids) -> int:
    """The known-truth self-check: the planted style effect must be recoverable."""
    style = pd.Series(rater_style, index=rater_ids)
    s = df.assign(style=style.reindex(df.rater_id.astype(str)).to_numpy())
    r = s.groupby("style", observed=True).response_emotion.value_counts(normalize=True).unstack()
    d_fear = r.loc[1, "fear"] - r.loc[0, "fear"]
    d_sad = r.loc[1, "sad"] - r.loc[0, "sad"]
    assert d_fear > 0.02, f"planted fear tilt not recoverable: {d_fear:.4f}"
    assert d_sad < -0.02, f"planted sad tilt not recoverable: {d_sad:.4f}"
    hit = (df.response_emotion == df.intended_emotion).mean()
    assert 0.35 < hit < 0.75, f"fixture legibility {hit:.3f} is not CREMA-D-plausible"
    assert df.groupby("clip_id", observed=True).size().min() >= 1
    print(f"ok  planted DIF recoverable (fear {d_fear:+.3f}, sad {d_sad:+.3f}), "
          f"P(response==intended)={hit:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
