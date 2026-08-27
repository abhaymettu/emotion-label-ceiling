"""Fine-tune wav2vec2/WavLM on CREMA-D audio for 6-class emotion, split by ACTOR.

Everything a run produces lands in modeling/runs/<tag>/ so a killed session loses nothing:
  config.json   exact args, seed, split, package versions
  curves.csv    per-epoch train loss / val accuracy
  metrics.json  test accuracy vs intended AND vs audio consensus, per-class, ceiling comparison
  confusion.csv 6x6 confusion matrix on the test actors
Weights go to models/ (gitignored).

  .venv/bin/python modeling/finetune.py --tag w2v2-intended-s0
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, EMOTIONS, E2I, SR, MAX_SAMPLES, load_clips, load_wave,
                    actor_split, random_clip_split, split_frames)


class Clips(Dataset):
    def __init__(self, df, label_col):
        self.paths = df.path.tolist()
        self.y = [E2I[e] for e in df[label_col]]
        self.intended = [E2I[e] for e in df.intended_emotion]
        self.cons = [E2I[e] for e in df.consensus_audio]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_wave(self.paths[i]), self.y[i], self.intended[i], self.cons[i]


def make_collate(pad_to=None):
    """pad_to=None pads each batch to its own longest clip. pad_to=MAX_SAMPLES pads every
    batch to the same width.

    This is not a cosmetic choice on MPS. Per-batch padding gives nearly every step a
    unique tensor shape, and each new shape compiles a fresh MPSGraph that is then cached
    forever. With wav2vec2-large the cache of 24-layer graphs grows until the machine
    swaps: measured 0.90 s/step at step 40 and 19.48 s/step at step 100, a 20x collapse,
    extrapolating to 199 min/epoch. Padding to one fixed width holds it flat at 0.95
    s/step, 10 min/epoch. Fixed padding does change what the encoder attends over, so runs
    with and without it are not comparable to each other -- hence the -fixpad tag."""
    def collate(batch):
        n = pad_to or max(len(b[0]) for b in batch)
        x = torch.zeros(len(batch), n)
        m = torch.zeros(len(batch), n)
        for i, (w, *_) in enumerate(batch):
            x[i, :len(w)] = torch.from_numpy(w)
            m[i, :len(w)] = 1
        cols = [torch.tensor([b[j] for b in batch]) for j in (1, 2, 3)]
        return (x, m, *cols)
    return collate


collate = make_collate()


class Model(nn.Module):
    def __init__(self, name, n_cls=6, freeze_cnn=True):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        if freeze_cnn and hasattr(self.enc, "feature_extractor"):
            self.enc.feature_extractor._freeze_parameters()
        self.head = nn.Linear(self.enc.config.hidden_size, n_cls)

    def pooled(self, x, m):
        h = self.enc(x).last_hidden_state
        # mask down to the real frames; the CNN stride is ~320 samples
        f = torch.nn.functional.interpolate(m[:, None], size=h.shape[1], mode="nearest")[:, 0]
        return (h * f[..., None]).sum(1) / f.sum(1, keepdim=True).clamp(min=1)

    def forward(self, x, m):
        return self.head(self.pooled(x, m))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    P, Y, I, C = [], [], [], []
    for x, m, y, i_, c in loader:
        p = model(x.to(dev), m.to(dev)).argmax(-1).cpu()
        P.append(p); Y.append(y); I.append(i_); C.append(c)
    return [torch.cat(v).numpy() for v in (P, Y, I, C)]


def per_class(pred, targ):
    out = {}
    for e, i in E2I.items():
        s = targ == i
        out[e] = {"n": int(s.sum()), "recall": float((pred[s] == i).mean()) if s.any() else None,
                  "precision": float((targ[pred == i] == i).mean()) if (pred == i).any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--label", default="intended_emotion",
                    choices=["intended_emotion", "consensus_audio"])
    ap.add_argument("--split", default="actor", choices=["actor", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--pad", default="dynamic", choices=["dynamic", "fixed"],
                    help="fixed pads every batch to MAX_SAMPLES; see make_collate")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    suffix = "-fixpad" if a.pad == "fixed" else ""
    tag = a.tag or f"{a.model.split('/')[-1]}-{a.label}-{a.split}{suffix}-s{a.seed}"
    out = ROOT / "modeling/runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    clips = load_clips()
    if a.split == "actor":
        sp = actor_split(clips, seed=a.seed)
        f = split_frames(clips, sp)
        split_desc = {k: sorted(v) for k, v in sp.items()}
    else:
        sp = random_clip_split(clips, seed=a.seed)
        f = {k: clips[clips.clip_id.isin(v)].reset_index(drop=True) for k, v in sp.items()}
        split_desc = "random clip split (cautionary baseline, actors leak across sides)"
    assert set(f["train"].actor_id) & set(f["test"].actor_id) == set() or a.split == "random"

    json.dump({**vars(a), "tag": tag, "device": dev, "torch": torch.__version__,
               "n_train": len(f["train"]), "n_val": len(f["val"]), "n_test": len(f["test"]),
               "n_clips_available": len(clips), "split": split_desc,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out / "config.json", "w"), indent=2)

    # num_workers=0 on purpose. Forking after MPS/Metal is initialised deadlocks the
    # workers on macOS -- intermittently, which looks like a very slow epoch rather than
    # a hang. Reading the whole 4,906-clip epoch off disk costs ~2.5s, so workers bought
    # nothing anyway. Batch order is unaffected: the sampler runs in the main process.
    dl = {k: DataLoader(Clips(v, a.label), batch_size=a.bs, shuffle=(k == "train"),
                        collate_fn=make_collate(MAX_SAMPLES if a.pad == "fixed" else None),
                        num_workers=0)
          for k, v in f.items()}
    model = Model(a.model).to(dev)
    # record what encoder actually got built, so a config difference cannot hide
    ec = model.enc.config
    cfg = json.load(open(out / "config.json"))
    cfg["encoder"] = {"hidden_size": ec.hidden_size, "num_hidden_layers": ec.num_hidden_layers,
                      "num_attention_heads": ec.num_attention_heads, "layerdrop": ec.layerdrop,
                      "hidden_dropout": ec.hidden_dropout, "mask_time_prob": ec.mask_time_prob,
                      "do_stable_layer_norm": ec.do_stable_layer_norm,
                      "feat_extract_norm": ec.feat_extract_norm,
                      "enc_params": sum(p.numel() for p in model.enc.parameters()),
                      "total_params": sum(p.numel() for p in model.parameters())}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.head_lr}])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.head_lr], total_steps=a.epochs * len(dl["train"]), pct_start=0.1)
    lossf = nn.CrossEntropyLoss()

    curves, best = [], -1
    for ep in range(a.epochs):
        model.train(); tot = n = 0; t0 = time.time()
        for x, m, y, *_ in dl["train"]:
            opt.zero_grad()
            loss = lossf(model(x.to(dev), m.to(dev)), y.to(dev))
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        P, Y, _, _ = evaluate(model, dl["val"], dev)
        acc = float((P == Y).mean())
        curves.append({"epoch": ep, "train_loss": tot / n, "val_acc": acc,
                       "secs": round(time.time() - t0, 1)})
        pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
        print(f"ep{ep} loss {tot/n:.4f} val_acc {acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            (ROOT / "models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "models" / f"{tag}.pt")

    model.load_state_dict(torch.load(ROOT / "models" / f"{tag}.pt"))
    P, Y, I, C = evaluate(model, dl["test"], dev)
    cm = pd.crosstab(pd.Series([EMOTIONS[i] for i in Y], name="target"),
                     pd.Series([EMOTIONS[i] for i in P], name="pred"))
    cm.to_csv(out / "confusion.csv")
    pd.DataFrame({"clip_id": f["test"].clip_id, "pred": [EMOTIONS[i] for i in P],
                  "intended": [EMOTIONS[i] for i in I],
                  "consensus_audio": [EMOTIONS[i] for i in C]}).to_csv(out / "test_preds.csv", index=False)

    ceil = {}
    cj = ROOT / "ceiling/out/ceiling.json"
    if cj.exists():
        c = json.load(open(cj))["by_modality"]["audio"]
        ceil = {"audio_ceiling_panel10": c["ceiling_headline"],
                "crowd_consensus_vs_intended": c["consensus_vs_intended_accuracy"],
                "krippendorff_alpha_audio": c["reliability"]["krippendorff_alpha_nominal"]}

    m = {"tag": tag, "n_test_clips": int(len(Y)), "n_test_actors": int(f["test"].actor_id.nunique()),
         "n_seeds": 1, "seed": a.seed, "split": a.split, "trained_against": a.label, "pad": a.pad,
         "best_val_acc": best,
         "test_acc_vs_intended": float((P == I).mean()),
         "test_acc_vs_audio_consensus": float((P == C).mean()),
         "per_class_vs_intended": per_class(P, I),
         "per_class_vs_audio_consensus": per_class(P, C),
         "ceiling_context": ceil,
         "caveat": "single run, single seed. one run is not a result; "
                   "re-run with --seed 1,2 before reporting a spread.",
         "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if ceil:
        m["headroom_to_audio_ceiling"] = ceil["audio_ceiling_panel10"]["estimate"] - m["test_acc_vs_audio_consensus"]
    json.dump(m, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps({k: m[k] for k in ("test_acc_vs_intended", "test_acc_vs_audio_consensus")}, indent=2))


if __name__ == "__main__":
    main()
