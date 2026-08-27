"""Build web/index.html: a self-contained page with real CREMA-D clips baked in.

Picks clips that span the agreement range (a couple the crowd nailed, most where it
split), copies them to demo/, and inlines them as base64 so the page works from disk
with no server and no CDN. If a finished training run has dumped test_preds.csv the
model's guess is folded in; if not, the page says so rather than inventing one.

  .venv/bin/python web/build.py [--run <tag>] [--n 12]
"""
import argparse, base64, json, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modeling"))
from common import EMOTIONS, load_clips, audio_dir

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None, help="modeling/runs/<tag> to take model predictions from")
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()

clips = load_clips()
ratings = pd.read_parquet(ROOT / "data/ratings_long.parquet")
aud = ratings[ratings.presented_modality == "audio"]

# only clips whose test-set membership we can honour: if a run exists, restrict to its
# test actors so the model prediction shown is genuinely held out.
preds = {}
run_meta = None
if a.run:
    rd = ROOT / "modeling/runs" / a.run
    tp = rd / "test_preds.csv"
    if tp.exists():
        p = pd.read_csv(tp)
        preds = dict(zip(p.clip_id, p.pred))
        clips = clips[clips.clip_id.isin(preds)]
        run_meta = json.load(open(rd / "metrics.json")) if (rd / "metrics.json").exists() else {"tag": a.run}
    else:
        print(f"no test_preds.csv in {rd}; building page without model predictions")

pool = clips[clips.n_ratings_audio >= 9].copy()
# spread across the agreement range, and across all six intended emotions
pool = pool.sort_values("agreement_audio")
picks = []
for e in EMOTIONS:
    g = pool[pool.intended_emotion == e]
    if len(g) == 0:
        continue
    picks.append(g.iloc[len(g) // 6])          # a contested one
    picks.append(g.iloc[int(len(g) * 0.92)])   # one the crowd mostly agreed on
picks = pd.DataFrame(picks).drop_duplicates("clip_id").head(a.n)

demo = ROOT / "demo"
demo.mkdir(exist_ok=True)
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
}

tpl = (ROOT / "web/template.html").read_text()
out = tpl.replace("/*__DATA__*/", json.dumps(payload))
(ROOT / "web/index.html").write_text(out)
kb = len(out) / 1024
print(f"wrote web/index.html ({kb:.0f} KB, {len(items)} clips, "
      f"model preds: {'yes' if preds else 'no'})")
