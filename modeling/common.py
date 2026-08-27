"""Shared modeling bits: label vocab, audio loading, and the actor-disjoint split.

The split is the whole point of this file. A random clip split puts the same actor's
voice on both sides and the model learns the speaker, not the emotion. Everything here
splits by actor. The random split exists only so we can report what it would have given.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
E2I = {e: i for i, e in enumerate(EMOTIONS)}
SR = 16000
MAX_SAMPLES = 4 * SR  # p95 clip is 3.64s; 4s covers all but a handful, which get trimmed


def audio_dir():
    for p in [ROOT / "data/audio/repo/AudioWAV", ROOT / "data/raw/AudioWAV",
              ROOT / "data/raw/repo/AudioWAV"]:
        if p.is_dir():
            return p
    raise FileNotFoundError("no AudioWAV directory found under data/")


def load_clips(require_audio=True):
    c = pd.read_parquet(ROOT / "data/clips.parquet")
    d = audio_dir()
    c["path"] = [str(d / w) for w in c.wav_file]
    if require_audio:
        have = np.array([Path(p).exists() for p in c.path])
        if not have.all():
            print(f"WARNING: {(~have).sum()} of {len(c)} wav files missing; using {have.sum()}")
        c = c[have].reset_index(drop=True)
    return c


def load_wave(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    if x.ndim > 1:
        x = x.mean(1)
    return x[:MAX_SAMPLES]


def actor_split(clips, seed=0, n_val=11, n_test=20):
    """Actor-disjoint train/val/test. Stratified by actor sex so the held-out panel
    is not accidentally all one voice type."""
    actors = clips[["actor_id", "actor_sex"]].drop_duplicates().sort_values("actor_id")
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    frac_v, frac_t = n_val / len(actors), n_test / len(actors)
    for _, g in actors.groupby("actor_sex", observed=True):
        ids = np.sort(g.actor_id.to_numpy())
        rng.shuffle(ids)
        kv = int(round(frac_v * len(ids)))
        kt = int(round(frac_t * len(ids)))
        val += list(ids[:kv])
        test += list(ids[kv:kv + kt])
        train += list(ids[kv + kt:])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def random_clip_split(clips, seed=0, n_val=11, n_test=20):
    """CAUTIONARY BASELINE ONLY. Splits clips at random, so actors straddle the split.
    Never report a number from this as the headline."""
    n = len(clips)
    per_actor = n / clips.actor_id.nunique()
    kv, kt = int(round(n_val * per_actor)), int(round(n_test * per_actor))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return {"val": clips.clip_id.to_numpy()[idx[:kv]],
            "test": clips.clip_id.to_numpy()[idx[kv:kv + kt]],
            "train": clips.clip_id.to_numpy()[idx[kv + kt:]]}


def split_frames(clips, split):
    """split: dict of actor_id lists -> dict of clip DataFrames."""
    return {k: clips[clips.actor_id.isin(v)].reset_index(drop=True) for k, v in split.items()}


def _selfcheck():
    clips = load_clips(require_audio=False)
    for seed in range(5):
        sp = actor_split(clips, seed=seed)
        tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
        assert tr & va == set() and tr & te == set() and va & te == set(), \
            f"actor sets overlap at seed {seed}"
        assert tr | va | te == set(clips.actor_id), "split lost or invented an actor"
        f = split_frames(clips, sp)
        cid = {k: set(v.clip_id) for k, v in f.items()}
        assert cid["train"] & cid["val"] == set() and cid["train"] & cid["test"] == set() \
            and cid["val"] & cid["test"] == set(), "clip ids overlap"
        assert sum(len(v) for v in cid.values()) == len(clips), "clips lost"
        # the real property we care about: no actor's voice appears on both sides
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert set(f[a].actor_id) & set(f[b].actor_id) == set(), \
                f"actor leaked between {a} and {b}"
        print(f"seed {seed}: actors {len(tr)}/{len(va)}/{len(te)}  "
              f"clips {len(f['train'])}/{len(f['val'])}/{len(f['test'])}  disjoint OK")
    # and confirm the random split does NOT have that property, which is the point
    rs = random_clip_split(clips, seed=0)
    a_tr = set(clips[clips.clip_id.isin(rs["train"])].actor_id)
    a_te = set(clips[clips.clip_id.isin(rs["test"])].actor_id)
    assert a_tr & a_te, "random split unexpectedly actor-disjoint; the contrast is broken"
    print(f"random split leaks {len(a_tr & a_te)} actors across train/test, as expected")
    print("split self-check passed")


if __name__ == "__main__":
    _selfcheck()
