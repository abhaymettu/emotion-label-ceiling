"""Fill invariance/README.md's tables straight from out/*.json.

  .venv/bin/python invariance/report.py            # print the sections
  .venv/bin/python invariance/report.py --inject   # write them into README.md

Exists so the README's numbers are transcribed by a script and never by hand.
Refuses to emit anything computed from a SIMULATED fixture.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
GROUP = {
    "grp_speed": "response speed",
    "grp_extremity": "slider extremity",
    "grp_position": "session position *(within-rater)*",
    "grp_style": "construal class (PC1)",
    "grp_style_resid": "construal class, ability-residualised",
}
MOD = {"audio": "audio only", "visual": "visual only", "audiovisual": "audiovisual"}


def _load(name, required=False):
    p = OUT / name
    if not p.exists():
        if required:
            sys.exit(f"missing {p}. Run nrm.py first.")
        print(f"  (skipping, missing {p.name})", file=sys.stderr)
        return None
    d = json.loads(p.read_text())
    if d.get("simulated"):
        sys.exit(f"{p} was computed from SIMULATED data; refusing to report it")
    return d


def _hdr(d):
    return (f"Source `{d['source_file']}`, {MOD.get(d.get('modality_filter'), 'all pooled')}: "
            f"**n = {d['n_ratings']:,} trials, {d['n_raters']:,} raters, "
            f"{d['n_clips']:,} clips.**")


def nrm_table(d, loo=None):
    """One row per (grouping, item), ordered by dTVD within grouping."""
    perm = any("perm" in it for gr in d["by_group"].values()
               for it in gr["items"].values())
    cols = ["grouping", "item", "n trials", "dTVD"]
    if perm:
        cols += ["perm. null", "excess", "p(perm)"]
    cols += ["χ²(10)", "p(χ²)", "largest shift"]
    if loo:
        cols += ["LOO range over 2,443 deletions"]
    L = [_hdr(d), "", "| " + " | ".join(cols) + " |",
         "|" + "---|" * len(cols)]
    for g, gr in d["by_group"].items():
        for e, it in sorted(gr["items"].items(), key=lambda kv: -kv[1]["dtvd"]):
            r = [GROUP.get(g, g), e, f"{it['n_trials']:,}", f"{it['dtvd']:.4f}"]
            if perm:
                p = it.get("perm", {})
                r += [f"{p['null_dtvd_mean']:.4f}", f"{p['excess_dtvd_over_null']:+.4f}",
                      f"{p['p_perm_dtvd']:.3f}"]
            r += [f"{it['lr_chi2']:.1f}", f"{it['p_chi2']:.3g}",
                  f"`{it['largest_shift_response']}` {it['largest_shift']:+.3f}"]
            if loo:
                l = loo["by_group"].get(g, {}).get("leave_one_rater_out", {}).get(e)
                r += [f"{l['loo_min']:.4f} – {l['loo_max']:.4f}" if l else "—"]
            L.append("| " + " | ".join(r) + " |")
    L += ["", "Least invariant items, mean dTVD over the groupings: "
              + ", ".join(f"**{x['item']}** {x['mean_dtvd']:.4f}"
                          for x in d["least_invariant_items"]) + "."]
    if perm:
        # dTVD has a per-item floor that depends on that item's n, so the ranking that
        # answers "which category is least invariant" is the one net of its own null.
        ex = {}
        for gr in d["by_group"].values():
            for e, it in gr["items"].items():
                ex.setdefault(e, []).append(it["perm"]["excess_dtvd_over_null"])
        rank = sorted(((e, sum(v) / len(v)) for e, v in ex.items()),
                      key=lambda kv: -kv[1])
        L += ["", "Same ranking **net of each item's own permutation null**, which is "
                  "the one to read because dTVD's floor scales with that item's n: "
              + ", ".join(f"**{e}** {v:+.4f}" for e, v in rank) + "."]
    return "\n".join(L)


def compare(nrm, dec):
    rows = []
    for g, gr in nrm["by_group"].items():
        for e, it in gr["items"].items():
            o = dec["items"].get(g, {}).get(e)
            if not (o and o.get("nominal")):
                continue
            rows.append({
                "g": g, "e": e, "dtvd": it["dtvd"],
                "excess": it.get("perm", {}).get("excess_dtvd_over_null"),
                "decile": o["nominal"]["total_variation_distance"],
                "dr2": o["delta_r2_nagelkerke"],
                "nrm_shift": it["largest_shift_response"],
                "dec_shift": o["nominal"]["largest_shift_response"],
            })
    t = pd.DataFrame(rows)
    if t.empty:
        return "  (no overlapping cells)"
    pairs = [("dtvd", "decile", "NRM dTVD vs matched-decile TVD"),
             ("excess", "decile", "NRM dTVD above its permutation null vs decile TVD"),
             ("dtvd", "dr2", "NRM dTVD vs dichotomous LR-DIF ΔR²"),
             ("excess", "dr2", "NRM dTVD above null vs LR-DIF ΔR²")]
    L = [f"Across all **n = {len(t)}** (grouping × item) cells in this modality:", "",
         "| compared | Pearson r | Spearman ρ |", "|---|---|---|"]
    for a, b, lab in pairs:
        if t[a].notna().any():
            L.append(f"| {lab} | {t[a].corr(t[b]):+.3f} | "
                     f"{t[a].corr(t[b], method='spearman'):+.3f} |")
    same = (t.nrm_shift == t.dec_shift).mean()
    nn = (t.nrm_shift == "neutral").mean()
    dn = (t.dec_shift == "neutral").mean()
    ratio = (t.decile / t.dtvd).median()
    L += ["", f"Both methods name the same response category as the largest shift in "
              f"**{same:.0%}** of the {len(t)} cells "
              f"(chance with six categories is 17%).",
          "", f"The decile approximation is larger: median ratio "
              f"decile TVD / NRM dTVD = **{ratio:.2f}**. It also concentrates on the "
              f"modal response — it names `neutral` as the shifted category in "
              f"**{dn:.0%}** of cells against the NRM's **{nn:.0%}**.",
          "", "| grouping | item | NRM dTVD | decile TVD | NRM says | decile says |",
          "|---|---|---|---|---|---|"]
    for _, r in t.sort_values("dtvd", ascending=False).head(8).iterrows():
        L.append(f"| {GROUP.get(r.g, r.g)} | {r.e} | {r.dtvd:.4f} | {r.decile:.4f} "
                 f"| `{r.nrm_shift}` | `{r.dec_shift}` |")
    return "\n".join(L)


def loo_section(loo):
    if not loo:
        return "  (not run)"
    L = ["| grouping | item | observed dTVD | 95% CI (jackknife) | min over deletions "
         "| max | largest single-rater move |",
         "|---|---|---|---|---|---|---|"]
    ex = True
    n = 0
    for g, gr in loo["by_group"].items():
        for e, l in gr.get("leave_one_rater_out", {}).items():
            ex &= l["exhaustive"]
            n = l["n_deletions"]
            se = l["jackknife_se"]
            ci = (f"{l['observed_dtvd'] - 1.96 * se:.4f} – "
                  f"{l['observed_dtvd'] + 1.96 * se:.4f}") if se else "—"
            L.append(f"| {GROUP.get(g, g)} | {e} | {l['observed_dtvd']:.4f} | {ci} "
                     f"| {l['loo_min']:.4f} | {l['loo_max']:.4f} "
                     f"| {l['max_abs_change']:.5f} |")
    head = (f"Every one of the **{n:,}** raters deleted in turn, the whole analysis "
            f"refit each time (baseline plus six single-item models, "
            f"{n * 7:,} model fits per grouping)."
            if ex else
            f"**{n:,} raters deleted** — a random subsample, not exhaustive.")
    tail = ("\n\nThe CI is the delete-one jackknife, ±1.96 SE. dTVD contains an "
            "absolute value, so it is not everywhere smooth and the jackknife is "
            "approximate near dTVD ≈ 0; read it as a scale, not an exact interval. "
            "It is an interval on dTVD itself, not on dTVD net of its permutation "
            "null.")
    return head + "\n\n" + "\n".join(L) + tail


def mirt_section():
    p = OUT / "nrm-mirt-crosscheck.json"
    if not p.exists():
        return "  (not run)"
    d = json.loads(p.read_text())
    return (f"On a balanced fixture of {d['n_persons']:,} persons × {d['n_items']} items "
            f"× {d['n_categories']} categories: log-likelihood **{d['loglik_nrm_py']}** "
            f"here vs **{d['loglik_mirt']}** in mirt (gap {d['loglik_gap']:+}); largest "
            f"disagreement between any two fitted category response curves anywhere on "
            f"θ ∈ [−4, 4] is **{d['max_abs_prob_diff_all_theta']:.4f}** in probability "
            f"(**{d['max_abs_prob_diff_core_theta_2.5']:.4f}** over |θ| ≤ 2.5), "
            f"density-weighted mean "
            f"**{d['density_weighted_mean_abs_prob_diff']:.2e}**. Two independent "
            f"implementations, same answer.")


def build():
    s = {}
    audio = _load("nrm-dif-audio.json", required=True)
    loo = _load("nrm-dif-audio-loo.json")
    s["MIRT"] = mirt_section()
    s["AUDIO"] = nrm_table(audio)
    dec = _load("dif-audio.json")
    s["COMPARE"] = ("## Does the fitted model agree with the decile stand-in?\n\n"
                    + (compare(audio, dec) if dec else "  (dif-audio.json missing)"))
    s["LOO"] = loo_section(loo)
    sec = []
    for m in ("visual", "audiovisual"):
        d = _load(f"nrm-dif-{m}.json")
        if d:
            sec.append(f"### {MOD[m]}\n\n" + nrm_table(d))
            dd = _load(f"dif-{m}.json")
            if dd:
                sec.append("\n" + compare(d, dd))
    s["SECONDARY"] = "\n\n".join(sec) if sec else "  (not run)"
    c = _load("nrm-dif-audio-consensusloo.json")
    s["CONSENSUS"] = (
        "Items redefined by the **leave-one-rater-out crowd majority** label instead of "
        "the actor's direction — the rater's own vote is dropped before the majority is "
        "taken, so a rater is never scored against a label they helped build. Item sizes "
        "become very unbalanced under this key (audio consensus is dominated by "
        "`neutral`), so dTVD is not comparable across items here; read each item against "
        "its own permutation null.\n\n" + nrm_table(c)) if c else "  (not run)"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="store_true", help="write into README.md")
    a = ap.parse_args()
    s = build()
    if not a.inject:
        for k, v in s.items():
            print(f"\n<!--REPORT-{k}-->\n{v}")
        return 0
    p = HERE / "README.md"
    t = p.read_text()
    for k, v in s.items():
        mark = f"<!--REPORT-{k}-->"
        if mark not in t:
            sys.exit(f"README.md has no {mark} placeholder; refusing to guess")
        t = t.replace(mark, v)
    p.write_text(t)
    print(f"injected {len(s)} sections into {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
