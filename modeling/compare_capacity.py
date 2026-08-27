"""Head-to-head: does more encoder capacity close the intent-vs-consensus gap?

Reads every modeling/runs/*/metrics.json and prints markdown tables. Numbers only
come from runs that finished on this machine; nothing is imputed.

  .venv/bin/python modeling/compare_capacity.py
"""
import json, sys, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


def load():
    runs = []
    for p in sorted((ROOT / "modeling/runs").glob("*/metrics.json")):
        m = json.load(open(p))
        if m.get("split") != "actor" or m.get("trained_against") != "intended_emotion":
            continue
        cfg = json.load(open(p.parent / "config.json"))
        m["model"] = cfg["model"].split("/")[-1]
        runs.append(m)
    return runs


def ms(xs):
    """mean, sd (sample). sd is None for n<2 -- one run is not a spread."""
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else None)


def fmt(x, n=4):
    return "--" if x is None else f"{x:.{n}f}"


def main():
    runs = load()
    if not runs:
        sys.exit("no finished actor-disjoint intent runs found")
    ceil = next((r["ceiling_context"]["audio_ceiling_panel10"]["estimate"]
                 for r in runs if r.get("ceiling_context")), None)
    models = sorted({r["model"] for r in runs},
                    key=lambda m: (0 if m.endswith("base") else 1, m))

    print("## Per seed\n")
    print("| model | seed | n test clips | vs intended | vs audio consensus | gap (pts) |")
    print("|---|---|---|---|---|---|")
    for mo in models:
        for r in sorted([x for x in runs if x["model"] == mo], key=lambda x: x["seed"]):
            i, c = r["test_acc_vs_intended"], r["test_acc_vs_audio_consensus"]
            print(f"| {mo} | {r['seed']} | {r['n_test_clips']} | {i:.4f} | {c:.4f} | {100*(i-c):.1f} |")

    print("\n## Means\n")
    print("| model | n seeds | vs intended (mean ± sd) | vs consensus (mean ± sd) "
          "| gap (pts) | headroom to 0.727 ceiling |")
    print("|---|---|---|---|---|---|")
    summ = {}
    for mo in models:
        rs = [x for x in runs if x["model"] == mo]
        mi, si = ms([r["test_acc_vs_intended"] for r in rs])
        mc, sc = ms([r["test_acc_vs_audio_consensus"] for r in rs])
        summ[mo] = (mi, si, mc, sc, len(rs))
        hd = f"{100*(ceil - mc):+.1f} pts below" if ceil else "--"
        print(f"| {mo} | {len(rs)} | {fmt(mi)} ± {fmt(si)} | {fmt(mc)} ± {fmt(sc)} "
              f"| {100*(mi-mc):.1f} | {hd} |")

    if len(models) >= 2:
        b, l = models[0], models[-1]
        mi_b, _, mc_b, _, nb = summ[b]
        mi_l, _, mc_l, _, nl = summ[l]
        print(f"\n**{l} minus {b}:** intended {100*(mi_l-mi_b):+.1f} pts, "
              f"consensus {100*(mc_l-mc_b):+.1f} pts, "
              f"gap {100*((mi_l-mc_l)-(mi_b-mc_b)):+.1f} pts "
              f"(n={nl} vs n={nb} seeds).")

    print("\n## Per class, vs audio consensus (mean over seeds)\n")
    print("| class | " + " | ".join(f"{mo} prec / rec" for mo in models) + " |")
    print("|---" * (len(models) + 1) + "|")
    for e in EMOTIONS:
        cells = []
        for mo in models:
            rs = [x for x in runs if x["model"] == mo]
            pr = [r["per_class_vs_audio_consensus"][e]["precision"] for r in rs]
            rc = [r["per_class_vs_audio_consensus"][e]["recall"] for r in rs]
            pr = [v for v in pr if v is not None]; rc = [v for v in rc if v is not None]
            cells.append(f"{fmt(st.mean(pr) if pr else None,3)} / {fmt(st.mean(rc) if rc else None,3)}")
        print(f"| {e} | " + " | ".join(cells) + " |")

    print("\n## Per class, vs intended (mean over seeds)\n")
    print("| class | " + " | ".join(f"{mo} prec / rec" for mo in models) + " |")
    print("|---" * (len(models) + 1) + "|")
    for e in EMOTIONS:
        cells = []
        for mo in models:
            rs = [x for x in runs if x["model"] == mo]
            pr = [r["per_class_vs_intended"][e]["precision"] for r in rs]
            rc = [r["per_class_vs_intended"][e]["recall"] for r in rs]
            pr = [v for v in pr if v is not None]; rc = [v for v in rc if v is not None]
            cells.append(f"{fmt(st.mean(pr) if pr else None,3)} / {fmt(st.mean(rc) if rc else None,3)}")
        print(f"| {e} | " + " | ".join(cells) + " |")

    for mo in models:
        n = len([x for x in runs if x["model"] == mo])
        if n < 2:
            print(f"\n> {mo}: n={n} run. One run is not a result -- no spread is claimed.")


def _selfcheck():
    assert ms([1.0, 2.0]) == (1.5, st.stdev([1.0, 2.0]))
    assert ms([1.0])[1] is None, "sd must be None at n=1, not 0"
    assert fmt(None) == "--"
    print("compare_capacity selfcheck passed")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
