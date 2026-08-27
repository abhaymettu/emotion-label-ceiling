"""Print the markdown tables that invariance/README.md quotes, straight from out/*.json.

  .venv/bin/python invariance/report.py

Exists so the README's numbers are transcribed by a script and not by a human.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
GROUP_LABEL = {
    "grp_speed": "response speed",
    "grp_extremity": "intensity-slider extremity",
    "grp_position": "session position (early/late)",
    "grp_style": "latent construal class (PC1)",
    "grp_style_resid": "latent class, ability-residualised",
}


def _load(name):
    p = OUT / name
    if not p.exists():
        print(f"  (missing: {p.name} — run nrm.py first)", file=sys.stderr)
        return None
    d = json.loads(p.read_text())
    assert not d["simulated"], f"{p} was computed from SIMULATED data"
    return d


def nrm_table(d, loo=None):
    print(f"n = {d['n_ratings']:,} trials, {d['n_raters']:,} raters, "
          f"{d['n_clips']:,} clips; source `{d['source_file']}`\n")
    print("| grouping | item | n trials | dTVD | perm. null mean | excess | p(perm) "
          "| LR chi2(10) | largest shift | " + ("LOO range |" if loo else ""))
    print("|---|---|---|---|---|---|---|---|---|" + ("---|" if loo else ""))
    for g, gr in d["by_group"].items():
        for e, it in sorted(gr["items"].items(), key=lambda kv: -kv[1]["dtvd"]):
            pm = it.get("perm", {})
            row = (f"| {GROUP_LABEL.get(g, g)} | {e} | {it['n_trials']:,} "
                   f"| {it['dtvd']:.4f} | "
                   f"{pm.get('null_dtvd_mean', float('nan')):.4f} | "
                   f"{pm.get('excess_dtvd_over_null', float('nan')):+.4f} | "
                   f"{pm.get('p_perm_dtvd', float('nan')):.3f} | "
                   f"{it['lr_chi2']:.1f} | {it['largest_shift_response']} "
                   f"{it['largest_shift']:+.3f} |")
            if loo:
                l = loo["by_group"].get(g, {}).get("leave_one_rater_out", {}).get(e)
                row += (f" {l['loo_min']:.4f}–{l['loo_max']:.4f} |" if l else " — |")
            print(row)
    print()


def agreement_table(nrm, dec):
    """Do the fitted NRM and the matched-decile approximation rank the same cells?"""
    rows = []
    for g, gr in nrm["by_group"].items():
        for e, it in gr["items"].items():
            d = dec["items"].get(g, {}).get(e)
            if d and d.get("nominal"):
                rows.append({
                    "group_var": g, "item": e,
                    "dtvd_nrm": it["dtvd"],
                    "excess_over_null": it.get("perm", {}).get("excess_dtvd_over_null"),
                    "decile_tvd_approx": d["nominal"]["total_variation_distance"],
                    "nrm_shift": it["largest_shift_response"],
                    "decile_shift": d["nominal"]["largest_shift_response"],
                    "lr_delta_r2": d["delta_r2_nagelkerke"],
                })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    print(f"n = {len(t)} (grouping x item) cells\n")
    print("| pair | Pearson r | Spearman rho |")
    print("|---|---|---|")
    for a, b, lab in [("dtvd_nrm", "decile_tvd_approx", "NRM dTVD vs decile TVD"),
                      ("excess_over_null", "decile_tvd_approx",
                       "NRM dTVD above its null vs decile TVD"),
                      ("dtvd_nrm", "lr_delta_r2", "NRM dTVD vs LR-DIF dR2"),
                      ("excess_over_null", "lr_delta_r2",
                       "NRM dTVD above its null vs LR-DIF dR2")]:
        if t[a].notna().any():
            print(f"| {lab} | {t[a].corr(t[b]):.3f} | "
                  f"{t[a].corr(t[b], method='spearman'):.3f} |")
    same = (t.nrm_shift == t.decile_shift).mean()
    print(f"\nSame category named as the largest shift by both: "
          f"{same:.0%} of {len(t)} cells.\n")
    return t


def main():
    for mod in ["audio", "visual", "audiovisual"]:
        d = _load(f"nrm-dif-{mod}.json")
        if not d:
            continue
        loo = _load(f"nrm-dif-{mod}-loo.json") if mod == "audio" else None
        print(f"\n### NRM DIF — {mod}\n")
        nrm_table(d, loo)
        print("least invariant items (mean dTVD over the five groupings): " +
              ", ".join(f"{r['item']} {r['mean_dtvd']:.4f}"
                        for r in d["least_invariant_items"]))
        dec = _load(f"dif-{mod}.json")
        if dec:
            print(f"\n### NRM vs the matched-decile approximation — {mod}\n")
            agreement_table(d, dec)
    for extra in ["nrm-dif-audio-consensusloo.json"]:
        d = _load(extra)
        if d:
            print(f"\n### Robustness: items keyed on the leave-one-rater-out "
                  f"consensus — audio\n")
            nrm_table(d)
    x = OUT / "nrm-mirt-crosscheck.json"
    if x.exists():
        print("\n### Cross-check against R/mirt\n")
        print("```json\n" + x.read_text().strip() + "\n```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
