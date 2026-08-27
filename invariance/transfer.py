"""Does rater non-invariance cost anything? The A -> B transfer test.

DIF says the items function differently across rater groups. It does not say
anyone should care. This puts a number on it.

f_A(clip) = the majority label of that clip among group-A raters. That is exactly
what a model trained to convergence on group-A labels would output, so the ceiling
of transfer can be measured without training an audio model at all.

The confound in any naive A/B split is panel size: half as many raters gives a
noisier consensus, and accuracy drops for that reason alone. The permutation null
-- random regroupings of the same sizes -- is matched on panel size by
construction, so the gap above it is attributable to the grouping.

  .venv/bin/python invariance/transfer.py --modality audio
  .venv/bin/python invariance/transfer.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ceiling"))
sys.path.insert(0, str(ROOT / "invariance"))
from common import EMOTIONS, E_IDX, K, load  # noqa: E402
from dif import add_groups  # noqa: E402

OUT = ROOT / "invariance" / "out"


def _consensus(counts: np.ndarray, rng) -> np.ndarray:
    """Argmax with uniform random tie-breaking; -1 where the panel is empty."""
    tot = counts.sum(1)
    u = rng.random(counts.shape) * (counts == counts.max(1, keepdims=True))
    return np.where(tot > 0, u.argmax(1), -1)


def _pack(df: pd.DataFrame):
    """Return (clip_index, response_index, rater_index, n_clips, n_raters)."""
    clips, ci = np.unique(df.clip_id.astype(str).to_numpy(), return_inverse=True)
    raters, ri = np.unique(df.rater_id.astype(str).to_numpy(), return_inverse=True)
    re_ = df.response_emotion.map(E_IDX).to_numpy()
    return ci, re_, ri, len(clips), len(raters)


def _counts_by_side(ci, re_, side, n_clips):
    """side is a bool per rating: True = group B. Returns (countsA, countsB)."""
    a = np.zeros((n_clips, K), np.int32)
    b = np.zeros((n_clips, K), np.int32)
    np.add.at(a, (ci[~side], re_[~side]), 1)
    np.add.at(b, (ci[side], re_[side]), 1)
    return a, b


def transfer_once(ci, re_, side, n_clips, rng, min_panel=3):
    """Accuracy of group-A consensus against group-B consensus, on clips where
    both sides have at least min_panel raters."""
    a, b = _counts_by_side(ci, re_, side, n_clips)
    ok = (a.sum(1) >= min_panel) & (b.sum(1) >= min_panel)
    if ok.sum() < 50:
        return np.nan, 0
    ya = _consensus(a[ok], rng)
    yb = _consensus(b[ok], rng)
    return float((ya == yb).mean()), int(ok.sum())


def run(df, meta, group_col, rng, n_perm=200, min_panel=3, n_boot=300) -> dict:
    d = df[df[group_col].notna()]
    ci, re_, ri, n_clips, n_raters = _pack(d)
    grp = d.groupby("rater_id", observed=True)[group_col].first()
    lv = sorted(grp.unique())
    if len(lv) != 2:
        return {}
    rater_is_b = (grp == lv[1])
    side = rater_is_b.reindex(d.rater_id.astype(str)).to_numpy()

    obs, n_used = transfer_once(ci, re_, side, n_clips, rng, min_panel)

    # permutation null: same group sizes, random membership. Matched on panel size.
    n_b = int(rater_is_b.sum())
    names = grp.index.to_numpy()
    lut = pd.Series(np.arange(len(names)), index=names)
    rid = lut.reindex(d.rater_id.astype(str)).to_numpy()
    null = []
    for _ in range(n_perm):
        mask = np.zeros(len(names), bool)
        mask[rng.choice(len(names), n_b, replace=False)] = True
        v, _ = transfer_once(ci, re_, mask[rid], n_clips, rng, min_panel)
        if not np.isnan(v):
            null.append(v)
    null = np.array(null)

    # clip bootstrap on the observed split
    a, b = _counts_by_side(ci, re_, side, n_clips)
    ok = np.flatnonzero((a.sum(1) >= min_panel) & (b.sum(1) >= min_panel))
    bs = []
    for _ in range(n_boot):
        s = rng.choice(ok, len(ok))
        bs.append((_consensus(a[s], rng) == _consensus(b[s], rng)).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])

    p = float((null <= obs).mean()) if len(null) else np.nan
    return {
        "group_var": group_col, "levels": [str(lv[0]), str(lv[1])],
        "min_panel_per_side": min_panel,
        "n_clips_used": n_used, "n_raters": int(n_raters),
        "n_raters_group_b": n_b,
        "transfer_accuracy": obs,
        "transfer_ci95": [float(lo), float(hi)],
        "permutation_null_mean": float(null.mean()) if len(null) else None,
        "permutation_null_ci95": [float(np.percentile(null, 2.5)),
                                  float(np.percentile(null, 97.5))] if len(null) else None,
        "degradation_vs_null": float(null.mean() - obs) if len(null) else None,
        "permutation_p_one_sided": p,
        "n_permutations": int(len(null)),
        "n_bootstrap": n_boot,
    }


def self_check() -> int:
    """Two planted worlds. Exchangeable raters -> no degradation. Two rater
    factions that systematically disagree -> degradation the null cannot explain."""
    rng = np.random.default_rng(0)
    n_clips, n_raters, per_clip = 1500, 400, 16

    def build(faction_strength):
        ci = np.repeat(np.arange(n_clips), per_clip)
        ri = rng.integers(0, n_raters, n_clips * per_clip)
        fac = rng.integers(0, 2, n_raters)
        truth = rng.integers(0, K, n_clips)
        p = np.full((len(ci), K), 0.10)
        p[np.arange(len(ci)), truth[ci]] = 0.50
        # factions push a different alternative on every clip
        alt = (truth[ci] + 1 + fac[ri]) % K
        p[np.arange(len(ci)), alt] += faction_strength
        p /= p.sum(1, keepdims=True)
        resp = (p.cumsum(1) > rng.random(len(ci))[:, None]).argmax(1)
        return ci, resp, fac[ri] == 1

    for strength, expect in ((0.0, "none"), (0.9, "big")):
        ci, resp, side = build(strength)
        obs, n = transfer_once(ci, resp, side, n_clips, rng)
        null = np.array([transfer_once(ci, resp,
                                       rng.permutation(side), n_clips, rng)[0]
                         for _ in range(60)])
        gap = null.mean() - obs
        if expect == "none":
            assert abs(gap) < 0.03, f"exchangeable raters should not degrade: gap={gap:.4f}"
        else:
            assert gap > 0.10, f"planted factions should degrade transfer: gap={gap:.4f}"
        print(f"  faction strength {strength}: transfer={obs:.3f} "
              f"null={null.mean():.3f} gap={gap:+.3f}  (n={n} clips)")

    # permuting rater labels must not change panel sizes -- the whole point
    ci, resp, side = build(0.9)
    a, b = _counts_by_side(ci, resp, side, n_clips)
    a2, b2 = _counts_by_side(ci, resp, rng.permutation(side), n_clips)
    assert abs(a.sum() - a2.sum()) / a.sum() < 0.02, "permutation changed panel sizes"

    print("ok  no false degradation under exchangeability, real degradation under factions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--modality", default=None)
    ap.add_argument("--min-panel", type=int, default=3)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return self_check()

    rng = np.random.default_rng(a.seed)
    df, meta = load(a.ratings)
    if a.modality:
        df = df[df.presented_modality == a.modality]
        meta = {**meta, "modality_filter": a.modality, "n_ratings": int(len(df))}
    df = add_groups(df, rng)

    res = {**meta, "min_panel_per_side": a.min_panel, "by_group": {}}
    for g in [c for c in df.columns if c.startswith("grp_")]:
        r = run(df, meta, g, rng, n_perm=a.permutations, min_panel=a.min_panel)
        if r:
            res["by_group"][g] = r

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"-{a.modality}" if a.modality else ""
    (OUT / f"transfer{suffix}.json").write_text(json.dumps(res, indent=2))

    tag = "  [SIMULATED]" if meta["simulated"] else ""
    print(f"\nsource: {res['source_file']}{tag}   modality={a.modality or 'all pooled'}")
    print(f"A -> B: group-A consensus scored against group-B consensus, "
          f">= {a.min_panel} raters per side\n")
    print(f"{'grouping':18s} {'transfer':>9s} {'95% CI':>16s} {'perm null':>10s} "
          f"{'gap':>7s} {'p':>7s} {'clips':>7s}")
    for g, r in res["by_group"].items():
        ci = f"[{r['transfer_ci95'][0]:.3f},{r['transfer_ci95'][1]:.3f}]"
        print(f"{g:18s} {r['transfer_accuracy']:9.3f} {ci:>16s} "
              f"{r['permutation_null_mean']:10.3f} {r['degradation_vs_null']:+7.3f} "
              f"{r['permutation_p_one_sided']:7.3f} {r['n_clips_used']:7,d}")
    print(f"\nwrote {OUT / f'transfer{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
