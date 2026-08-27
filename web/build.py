"""Build web/index.html: a self-contained page with real CREMA-D clips baked in.

Every clip on the page comes from the run's held-out test split, so the model answer
shown next to it is a genuine prediction on an actor the model never heard. The 12 are
picked to span the agreement range across all six emotions, half where the crowd's
majority matched the actor's script and half where it did not, and with the model's
hit rate inside the sample matched to its hit rate on the whole test set -- otherwise
the page would be a highlight reel. Audio is inlined as base64 so it works from disk
with no server and no CDN.

  .venv/bin/python web/build.py --run wav2vec2-base-intended_emotion-actor-s0

Without --run (or with a run that has no test_preds.csv) it builds the page with no
model column at all rather than inventing one.
"""
import argparse, base64, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modeling"))
from common import EMOTIONS, load_clips, audio_dir

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None, help="modeling/runs/<tag> to take model predictions from")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--seed", type=int, default=0, help="tie-break seed for clip selection")
a = ap.parse_args()

clips = load_clips()
ratings = pd.read_parquet(ROOT / "data/ratings_long.parquet")
aud = ratings[ratings.presented_modality == "audio"]

preds, run_meta, finding = {}, None, None
if a.run:
    rd = ROOT / "modeling/runs" / a.run
    tp = rd / "test_preds.csv"
    if tp.exists():
        p = pd.read_csv(tp)
        cfg = json.load(open(rd / "config.json"))
        # hard gate: a prediction on a clip the model trained on is meaningless.
        held_out = set(cfg["split"]["test"])
        seen = set(cfg["split"]["train"]) | set(cfg["split"]["val"])
        actor = p.clip_id.str.split("_").str[0]
        assert set(actor) <= held_out and not (set(actor) & seen), \
            "test_preds.csv contains clips from a training or validation actor"
        preds = dict(zip(p.clip_id, p.pred))
        clips = clips[clips.clip_id.isin(preds)]
        run_meta = json.load(open(rd / "metrics.json"))

        # the finding, counted off the full test split rather than off the 12 on screen
        div = p[p.consensus_audio != p.intended]
        finding = {
            "n_test": len(p),
            "n_actors": int(actor.nunique()),
            "acc_intended": float((p.pred == p.intended).mean()),
            "acc_consensus": float((p.pred == p.consensus_audio).mean()),
            "crowd_vs_intended": float((p.consensus_audio == p.intended).mean()),
            "n_diverged": len(div),
            "n_model_script": int((div.pred == div.intended).sum()),
            "n_model_crowd": int((div.pred == div.consensus_audio).sum()),
        }
        finding["n_model_neither"] = (finding["n_diverged"] - finding["n_model_script"]
                                      - finding["n_model_crowd"])
        assert abs(finding["acc_intended"] - run_meta["test_acc_vs_intended"]) < 1e-9
        assert abs(finding["acc_consensus"] - run_meta["test_acc_vs_audio_consensus"]) < 1e-9
    else:
        print(f"no test_preds.csv in {rd}; building page without model predictions")

pool = clips[clips.n_ratings_audio >= 9].copy()
pool["diverged"] = pool.consensus_audio != pool.intended_emotion

# One clip per emotion where the crowd's majority matched the actor's script, one where it
# did not. Inside the half where they disagreed, the model's allegiance (script / room /
# neither) is sampled in the same proportion it holds over the whole test split, and the
# model's hit rate over all 12 is matched to its hit rate over the whole test split. Without
# that the page is a highlight reel: picking on interestingness alone oversamples exactly the
# clips that flatter the claim.
n_half = a.n // 2
if preds:
    pool["model"] = pool.clip_id.map(preds)
    d = finding["n_diverged"]
    n_script = round(n_half * finding["n_model_script"] / d)
    n_room = round(n_half * finding["n_model_crowd"] / d)
    alleg = ["script"] * n_script + ["room"] * n_room
    alleg += ["neither"] * (n_half - len(alleg))
    n_agreed_wrong = max(0, round(a.n * (1 - finding["acc_intended"])) - n_room
                         - (n_half - n_script - n_room))
else:
    pool["model"] = pool.intended_emotion
    alleg, n_agreed_wrong = ["script"] * n_half, 0

def side(g):
    """model's allegiance on clips where the room and the script disagree"""
    return np.where(g.model == g.intended_emotion, "script",
                    np.where(g.model == g.consensus_audio, "room", "neither"))

rng = np.random.default_rng(a.seed)
# hand the scarce allegiances to emotions that can actually supply them
div = pool[pool.diverged].assign(side=lambda g: side(g))
supply = {e: set(div[div.intended_emotion == e].side) for e in EMOTIONS}
want = sorted(alleg, key=lambda s: sum(s in v for v in supply.values()))
free, plan = list(EMOTIONS), {}
for s in want:
    ok = [e for e in free if s in supply[e]]
    assert ok, f"no held-out emotion can supply a '{s}' clip"
    e = ok[int(rng.integers(len(ok)))]
    plan[e] = s
    free.remove(e)
agreed_wrong = set(rng.choice(EMOTIONS, size=n_agreed_wrong, replace=False).tolist())

picks, used = [], set()
for e in EMOTIONS:
    for diverged in (True, False):
        g = pool[(pool.intended_emotion == e) & (pool.diverged == diverged)]
        g = g[side(g) == plan[e]] if diverged else \
            g[(g.model == e) != (e in agreed_wrong)]
        assert len(g), f"no candidate for {e}, diverged={diverged}"
        # span the agreement range: contested slots aim low, agreed slots aim high
        target = g.agreement_audio.quantile(0.25 if diverged else 0.9)
        g = g.assign(_d=(g.agreement_audio - target).abs())
        # among the clips that hit the target, prefer an actor not already on the page
        g = g[g._d <= g._d.min() + 0.05]
        g = g.assign(_seen=g.actor_id.isin(used)).sort_values(["_seen", "_d", "clip_id"])
        used.add(g.iloc[0].actor_id)
        picks.append(g.iloc[0])
picks = pd.DataFrame(picks).drop_duplicates("clip_id")
assert len(picks) == a.n, f"picked {len(picks)} clips, wanted {a.n}"

demo = ROOT / "demo"
demo.mkdir(exist_ok=True)
for old in demo.glob("*.wav"):
    old.unlink()
items = []
for _, r in picks.iterrows():
    src = Path(r.path)
    shutil.copy(src, demo / src.name)
    votes = aud[aud.clip_id == r.clip_id].response_emotion.value_counts()
    items.append({
        "id": r.clip_id, "actor": r.actor_id, "sex": r.actor_sex, "age": int(r.actor_age),
        "sentence": r.sentence_id,
        "intended": r.intended_emotion,
        "consensus": r.consensus_audio,
        "agreement": round(float(r.agreement_audio), 3),
        "votes": {e: int(votes.get(e, 0)) for e in EMOTIONS},
        "n": int(votes.sum()),
        "model": preds.get(r.clip_id),
        "wav": base64.b64encode(src.read_bytes()).decode(),
    })

ceil = json.load(open(ROOT / "ceiling/out/ceiling.json"))["by_modality"]["audio"]
payload = {
    "clips": items,
    "emotions": EMOTIONS,
    "stats": {
        "n_clips": 7442, "n_raters": 2443, "n_actors": 91,
        "alpha": round(ceil["reliability"]["krippendorff_alpha_nominal"], 3),
        "pairwise": round(ceil["reliability"]["pairwise_percent_agreement"], 3),
        "consensus_vs_intended": round(ceil["consensus_vs_intended_accuracy"], 3),
        "ceiling": round(ceil["ceiling_headline"]["estimate"], 3),
        "ceiling_ci": [round(x, 3) for x in ceil["ceiling_headline"]["ci95"]],
    },
    "model": run_meta,
    "finding": finding,
}

tpl = (ROOT / "web/template.html").read_text()
out = tpl.replace("/*__DATA__*/", json.dumps(payload))
(ROOT / "web/index.html").write_text(out)
kb = len(out) / 1024
print(f"wrote web/index.html ({kb:.0f} KB, {len(items)} clips, "
      f"model preds: {'yes' if preds else 'no'})")
if preds:
    got = sum(i["model"] == i["intended"] for i in items)
    crowd = sum(i["consensus"] == i["intended"] for i in items)
    print(f"  in-sample: model {got}/{len(items)}, crowd majority {crowd}/{len(items)}; "
          f"full test split: model {finding['acc_intended']:.3f}, "
          f"crowd {finding['crowd_vs_intended']:.3f}")
