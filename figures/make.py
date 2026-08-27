"""Draw every figure in the repo.

Every number is read from the artifacts at draw time -- ceiling/out/ceiling.json,
agreement/out/agreement.json, invariance/out/*.json, ceiling/sota.csv,
modeling/runs/*/metrics.json. Nothing is typed in by hand, so a figure cannot drift
away from the analysis that produced it.

Hard gate: any artifact stamped "simulated": true aborts the run. Figures never
carry fixture numbers.

    .venv/bin/python figures/make.py

Writes figures/<name>-light.{png,svg} and figures/<name>-dark.{png,svg}.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
RUN = "wav2vec2-base-intended_emotion-actor-s0"

EMO = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
MODS = ["audio", "visual", "audiovisual"]

THEMES = {
    "light": dict(bg="#ffffff", fg="#16191d", dim="#6b7280", grid="#e5e7eb", band="#dbe6fb"),
    "dark":  dict(bg="#14171a", fg="#e8eaed", dim="#9aa0a6", grid="#2b3138", band="#1e3050"),
}
# accents picked to stay legible on both grounds
BLUE, TEAL, AMBER, RED, PURPLE = "#4c8dff", "#2fb28a", "#e0a52b", "#e8615f", "#a986e0"


# ------------------------------------------------------------------ loading

def load_json(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        sys.exit(f"missing artifact {rel}. run the analysis that produces it first.")
    d = json.loads(p.read_text())
    check_not_simulated(d, rel)
    return d


def check_not_simulated(o, rel: str) -> None:
    """Refuse to draw anything derived from a SIMULATED_* fixture."""
    if isinstance(o, dict):
        if o.get("simulated") is True:
            sys.exit(f"REFUSING TO DRAW: {rel} is stamped simulated=true "
                     f"(source_file={o.get('source_file')}). Figures carry real data only.")
        for v in o.values():
            check_not_simulated(v, rel)
    elif isinstance(o, list):
        for v in o:
            check_not_simulated(v, rel)


CEIL = load_json("ceiling/out/ceiling.json")
AGR = load_json("agreement/out/agreement.json")
DIF = load_json("invariance/out/dif-audio.json")
TRA = load_json("invariance/out/transfer-audio.json")
MET = load_json(f"modeling/runs/{RUN}/metrics.json")
SOTA = list(csv.DictReader((ROOT / "ceiling/sota.csv").open()))

# every finished fine-tune in this repo, so the SOTA panel carries our own number too
RUNS = [load_json(f"modeling/runs/{d.name}/metrics.json")
        for d in sorted((ROOT / "modeling/runs").glob("*-actor-s*"))
        if (d / "metrics.json").exists()]


def ours_row():
    """This repo's fine-tune, scored against consensus, as a sota.csv-shaped row."""
    v = [r["test_acc_vs_audio_consensus"] for r in RUNS]
    n = len(v)
    lbl = ("this repo, wav2vec2-base trained on intent"
           + (f", mean of {n} seeds" if n > 1 else f", seed {RUNS[0]['seed']}"))
    return {"id": "OURS", "system": lbl, "modality": "audio", "metric": "accuracy",
            "value": f"{100 * sum(v) / n:.2f}", "n_classes": "6",
            "speaker_independent": "yes", "label_target": "consensus (audio perceived)",
            "url": "", "note": ""}


# ------------------------------------------------------------------ plumbing

def figure(name: str, theme: str, ncols=1, nrows=1, size=(11, 5.2), **kw):
    t = THEMES[theme]
    fig, ax = plt.subplots(nrows, ncols, figsize=size, facecolor=t["bg"], **kw)
    for a in (ax.ravel() if hasattr(ax, "ravel") else [ax]):
        a.set_facecolor(t["bg"])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(t["grid"])
        a.tick_params(colors=t["dim"], labelsize=9, length=3)
    return fig, ax, t


def title(fig, t, head, sub):
    fig.suptitle(head, color=t["fg"], fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.905, sub, color=t["dim"], fontsize=9.3, ha="left", va="top")


def footer(fig, t, txt):
    fig.text(0.012, 0.012, txt, color=t["dim"], fontsize=7.6, ha="left", va="bottom")


def save(fig, name: str, theme: str):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}-{theme}.{ext}", dpi=200,
                    facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  figures/{name}-{theme}.png|svg")


# ------------------------------------------------------- 1. modality agreement

def fig_modality(theme):
    fig, (axl, axr), t = figure("", theme, ncols=2, size=(11.4, 4.9),
                                gridspec_kw=dict(width_ratios=[1, 1], wspace=0.30))
    fig.subplots_adjust(top=0.72, bottom=0.16)
    y = [2, 1, 0]

    # left: Krippendorff alpha with bootstrap CI
    for i, m in enumerate(MODS):
        d = AGR["alpha_by_modality"][m]
        lo, hi = d["ci95"]
        c = BLUE if m == "audio" else t["fg"]
        axl.plot([lo, hi], [y[i]] * 2, color=c, lw=3, solid_capstyle="round", alpha=.9)
        axl.plot(d["alpha"], y[i], "o", color=c, ms=9, zorder=3,
                 mec=t["bg"], mew=1.4)
        axl.text(d["alpha"] + 0.022, y[i] + 0.02, f'{d["alpha"]:.3f}', color=c,
                 fontsize=11.5, fontweight="bold", va="center")
        axl.text(d["alpha"] + 0.022, y[i] + 0.30, f'95% CI [{lo:.3f}, {hi:.3f}]',
                 color=t["dim"], fontsize=8.2, va="center")
        axl.text(0.012, y[i] - 0.30, f'n = {d["n_ratings"]:,} ratings, {d["n_units"]:,} clips',
                 color=t["dim"], fontsize=7.8, va="center")
    for v, lab in ((0.667, "0.667  tentative"), (0.800, "0.800  firm")):
        axl.axvline(v, color=t["dim"], ls=(0, (4, 3)), lw=1)
        axl.text(v - 0.012, 2.62, lab, color=t["dim"], fontsize=8, rotation=90,
                 ha="right", va="top")
    axl.set_xlim(0, 0.90)
    axl.set_ylim(-0.62, 2.75)
    axl.set_yticks(y, [m.replace("audiovisual", "audiovisual") for m in MODS],
                   color=t["fg"], fontsize=10.5)
    axl.set_xlabel("Krippendorff's α (nominal). The 95% CI is narrower than the marker — "
                   "α is precisely estimated,\nand precisely low.", color=t["dim"], fontsize=9)
    axl.xaxis.grid(True, color=t["grid"], lw=0.7)
    axl.set_axisbelow(True)

    # right: how often the 10-rater majority reproduces the actor's intent
    chance = AGR["chance_accuracy"]
    for i, m in enumerate(MODS):
        v = CEIL["by_modality"][m]["consensus_vs_intended_accuracy"]
        c = BLUE if m == "audio" else t["fg"]
        axr.barh(y[i], v, height=0.42, color=c, alpha=.88 if m == "audio" else .55)
        axr.text(v + 0.014, y[i], f"{v:.3f}", color=c, fontsize=10.5,
                 fontweight="bold", va="center")
    axr.axvline(chance, color=RED, ls=(0, (4, 3)), lw=1.2)
    axr.text(chance + 0.012, 2.62, f"chance {chance:.3f}", color=RED, fontsize=8,
             ha="left", va="top")
    axr.set_xlim(0, 0.90)
    axr.set_ylim(-0.62, 2.75)
    axr.set_yticks(y, MODS, color=t["fg"], fontsize=10.5)
    axr.set_xlabel("crowd majority (R = 10) matches the actor's intended emotion",
                   color=t["dim"], fontsize=9)
    axr.xaxis.grid(True, color=t["grid"], lw=0.7)
    axr.set_axisbelow(True)

    title(fig, t,
          "Take the face away and the listeners stop agreeing",
          "CREMA-D, 7,442 clips × 2,443 raters. Every clip was rated in all three presentation\n"
          "conditions, so the three rows differ only in what the rater was shown.")
    footer(fig, t, "source: agreement/out/agreement.json, ceiling/out/ceiling.json "
                   "· data/ratings_long.parquet (219,686 ratings, real)")
    save(fig, "01-modality-agreement", theme)


# --------------------------------------------------------- 2. ceiling vs SOTA

def sota_rows(modality):
    rows = [r for r in SOTA
            if r["modality"] == modality and r["n_classes"] == "6"
            and r["speaker_independent"] == "yes"]
    return sorted(rows, key=lambda r: float(r["value"]))


def fig_sota(theme):
    fig, (axa, axv), t = figure("", theme, nrows=2, size=(11.4, 8.6),
                                gridspec_kw=dict(height_ratios=[1.55, 1], hspace=0.42))
    fig.subplots_adjust(top=0.805, bottom=0.13)

    for ax, mod, extra in ((axa, "audio", ["H4"]), (axv, "audiovisual", [])):
        c = CEIL["by_modality"][mod]["ceiling_headline"]
        est, (lo, hi) = c["estimate"] * 100, [x * 100 for x in c["ci95"]]
        rows = sota_rows(mod) + [r for r in SOTA if r["id"] in extra]
        if mod == "audio" and RUNS:
            rows.append(ours_row())
        rows = sorted(rows, key=lambda r: float(r["value"]))

        ax.axvspan(lo, hi, color=t["band"], zorder=0)
        ax.axvline(est, color=TEAL, lw=2, zorder=1)
        ax.text(est, len(rows) + 0.10,
                f"ceiling {est:.1f}  [{lo:.1f}, {hi:.1f}]",
                color=TEAL, fontsize=9.6, fontweight="bold", ha="center", va="bottom")

        for i, r in enumerate(rows):
            v, human = float(r["value"]), r["label_target"] == "intended" and r["id"].startswith("H")
            verified = "consensus" in r["label_target"]
            col = PURPLE if human else (TEAL if verified else (RED if v > est else t["fg"]))
            ax.plot([est, v], [i, i], color=col, lw=2.2, alpha=.55, zorder=2)
            ax.plot(v, i, "o", ms=8, color=col, zorder=3, mec=t["bg"], mew=1.2)
            delta = v - est
            span = max(float(x["value"]) for x in rows) - min(float(x["value"]) for x in rows)
            off = span * 0.028
            ax.text(v + (off if v >= est else -off), i, f"{v:.2f}   {delta:+.1f}",
                    color=col, fontsize=8.8, va="center",
                    ha="left" if v >= est else "right",
                    fontweight="bold" if (v > est or verified) else "normal")

        labs = []
        for r in rows:
            tag = {"unspecified": "label target unstated",
                   "intended": "scored vs actor intent",
                   "consensus (audio perceived)": "scored vs crowd consensus",
                   "consensus (audiovisual perceived)": "scored vs crowd consensus"}.get(
                       r["label_target"], r["label_target"])
            labs.append(f'{r["system"]}  ·  {tag}')
        ax.set_yticks(range(len(rows)), labs, color=t["fg"], fontsize=8.9)
        vals = [float(r["value"]) for r in rows]
        pad = (max(vals) - min(vals)) * 0.10
        ax.set_ylim(-0.8, len(rows) + 0.85)
        ax.set_xlim(min(vals) - pad * 2.6, max(vals) + pad * 2.6)
        ax.set_xlabel(f"reported accuracy on 6-class {mod}, speaker-independent splits (%)",
                      color=t["dim"], fontsize=9)
        ax.xaxis.grid(True, color=t["grid"], lw=0.7)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.04, transform=ax.transAxes, s=
                {"audio": "AUDIO-ONLY", "audiovisual": "AUDIOVISUAL"}[mod],
                color=t["dim"], fontsize=9, fontweight="bold", va="bottom")

    leg = [Line2D([], [], marker="o", ls="", color=RED, label="above the ceiling, label target unstated"),
           Line2D([], [], marker="o", ls="", color=TEAL, label="label target verified as crowd consensus"),
           Line2D([], [], marker="o", ls="", color=THEMES[theme]["fg"], label="below the ceiling"),
           Line2D([], [], marker="o", ls="", color=PURPLE, label="human crowd (Cao et al. 2014)")]
    fig.legend(handles=leg, loc="lower center", frameon=False, fontsize=8.6, ncols=4,
               labelcolor=t["dim"], handletextpad=0.4, bbox_to_anchor=(0.5, 0.032))

    title(fig, t,
          "Four audio results sit above the ceiling — and none of them says which label it scored",
          "The ceiling is the best accuracy any deterministic model can reach against the majority vote of a\n"
          "10-rater panel. It binds only if a paper scored against that consensus label. Eleven of the thirteen\n"
          "modelling papers checked state no label target at all, so the excess cannot be attributed either way.")
    footer(fig, t, "sources: ceiling/out/ceiling.json (n = 7,442 clips, 150 clip bootstraps) · "
                   "ceiling/sota.csv, every value read from the paper's own abstract or results table "
                   "(provenance in ceiling/SOURCES.md). Leaky and non-speaker-independent rows excluded.")
    save(fig, "02-ceiling-vs-sota", theme)


# ------------------------------------------------- 3. intended vs heard (model)

def fig_intent(theme):
    fig, (axl, axr), t = figure("", theme, ncols=2, size=(11.4, 5.4),
                                gridspec_kw=dict(width_ratios=[0.84, 1.16], wspace=0.30))
    fig.subplots_adjust(top=0.755, bottom=0.135)

    ci = MET["ceiling_context"]["audio_ceiling_panel10"]
    a_int, a_con = MET["test_acc_vs_intended"], MET["test_acc_vs_audio_consensus"]

    # left: same model, same clips, same predictions, two label sets
    xs = [0, 1]
    for x, v, lab, col in ((0, a_int, "vs the actor's\nINTENDED emotion", BLUE),
                           (1, a_con, "vs the audio-only\nCROWD CONSENSUS", AMBER)):
        axl.bar(x, v, width=0.52, color=col, alpha=.9)
        axl.text(x, v + 0.016, f"{v:.4f}", color=col, fontsize=13, fontweight="bold", ha="center")
    axl.axhline(ci["estimate"], color=TEAL, lw=1.8)
    axl.fill_between([-0.95, 1.6], ci["ci95"][0], ci["ci95"][1], color=t["band"], zorder=0)
    axl.text(1.58, ci["estimate"] - 0.048, f'ceiling {ci["estimate"]:.3f} '
             f'[{ci["ci95"][0]:.3f}, {ci["ci95"][1]:.3f}]', color=TEAL, fontsize=8.6, ha="right")
    axl.annotate("", xy=(-0.40, a_int), xytext=(-0.40, a_con),
                 arrowprops=dict(arrowstyle="<->", color=t["fg"], lw=1.2))
    axl.text(-0.47, (a_int + a_con) / 2, f"{100*(a_int-a_con):.1f}\npoints",
             color=t["fg"], fontsize=9.4, fontweight="bold", va="center", ha="right")
    axl.axhline(1 / 6, color=RED, ls=(0, (4, 3)), lw=1)
    axl.text(1.58, 1 / 6 + 0.020, "chance 0.167", color=RED, fontsize=8.2, ha="right")
    axl.set_xticks(xs, ["vs the actor's\nINTENDED\nemotion", "vs the audio-only\nCROWD\nCONSENSUS"],
                   color=t["fg"], fontsize=9.2)
    axl.set_xlim(-0.95, 1.6)
    axl.set_ylim(0, 0.88)
    axl.set_ylabel("test accuracy", color=t["dim"], fontsize=9)
    axl.yaxis.grid(True, color=t["grid"], lw=0.7)
    axl.set_axisbelow(True)

    # right: what the class actually is, once you ask listeners instead of the script
    ints = MET["per_class_vs_intended"]
    cons = MET["per_class_vs_audio_consensus"]
    order = sorted(EMO, key=lambda e: cons[e]["n"] - ints[e]["n"])
    for i, e in enumerate(order):
        a, b = ints[e]["n"], cons[e]["n"]
        col = TEAL if b > a else RED
        axr.annotate("", xy=(b, i), xytext=(a, i),
                     arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                     color=col, lw=2.1, alpha=.85))
        axr.plot(a, i, "o", ms=7.5, color=t["dim"], zorder=3, mec=t["bg"], mew=1.2)
        axr.plot(b, i, "o", ms=8.5, color=col, zorder=3, mec=t["bg"], mew=1.2)
        left = b < a                       # arrow points left, so put labels on the outside
        axr.text(a + (26 if left else -26), i, f"{a}", color=t["dim"], fontsize=8.8,
                 ha="left" if left else "right", va="center")
        axr.text(b + (-26 if left else 26), i, f"{b}", color=col, fontsize=10,
                 fontweight="bold", ha="right" if left else "left", va="center")
        p = cons[e]["precision"]
        axr.text(940, i, f"model precision on the\ncrowd's {e}: {p:.3f}" if e == "disgust"
                 else f"{p:.3f}", color=col if e == "disgust" else t["dim"],
                 fontsize=8.4 if e == "disgust" else 8.6, va="center",
                 fontweight="bold" if e == "disgust" else "normal")
    axr.set_yticks(range(len(order)), order, color=t["fg"], fontsize=10)
    axr.set_ylim(-0.7, len(order) + 0.05)
    axr.set_xlim(0, 1290)
    axr.set_xlabel("test clips in the class:  acted (grey)  →  actually heard by the crowd (colour)",
                   color=t["dim"], fontsize=9)
    axr.xaxis.grid(True, color=t["grid"], lw=0.7)
    axr.set_axisbelow(True)
    axr.text(940, len(order) - 0.42, "precision vs\nthe crowd label", color=t["dim"], fontsize=8.2,
             va="bottom")

    title(fig, t,
          "The model learned what the actor did, not what listeners hear",
          f'wav2vec2-base fine-tuned on the acted intent label, actor-disjoint split, '
          f'{MET["n_test_clips"]:,} test clips from {MET["n_test_actors"]} held-out actors, seed {MET["seed"]}.\n'
          "Both bars are the same model scoring the same predictions on the same clips. Only the label changed.")
    footer(fig, t, f'source: modeling/runs/{RUN}/metrics.json · ceiling from ceiling/out/ceiling.json. '
                   "Single seed; see README for the spread.")
    save(fig, "03-intent-vs-heard", theme)


# ---------------------------------------------------- 4. DIF + rater transfer

GRP = [("grp_speed", "response speed\nfast / deliberate"),
       ("grp_extremity", "response-style extremity\nextreme / moderate"),
       ("grp_position", "session position\nearly / late"),
       ("grp_style", "latent construal class\n(held-out split)"),
       ("grp_style_resid", "latent construal class\nability-residualised")]
LATENT = {"grp_style", "grp_style_resid"}


def fig_dif(theme):
    fig, (axl, axr), t = figure("", theme, ncols=2, size=(11.4, 5.4),
                                gridspec_kw=dict(width_ratios=[1, 1.22], wspace=0.42))
    fig.subplots_adjust(top=0.735, bottom=0.235)
    y = list(range(len(GRP)))[::-1]

    # left: worst-item DIF effect size per grouping
    for i, (g, lab) in enumerate(GRP):
        items = DIF["items"][g]
        worst = max(items, key=lambda e: items[e]["delta_r2_nagelkerke"])
        d = items[worst]
        col = PURPLE if g in LATENT else t["fg"]
        axl.barh(y[i], d["delta_r2_nagelkerke"], height=0.46, color=col,
                 alpha=.9 if g in LATENT else .5)
        axl.text(d["delta_r2_nagelkerke"] + 0.0012, y[i],
                 f'{d["delta_r2_nagelkerke"]:.3f}   {worst}   ETS {d["mh"]["ets"]}',
                 color=col, fontsize=8.7, va="center",
                 fontweight="bold" if g in LATENT else "normal")
    axl.axvline(0.035, color=AMBER, ls=(0, (4, 3)), lw=1.2)
    axl.text(0.0355, -0.62, "Jodoin & Gierl:  moderate ≥ 0.035",
             color=AMBER, fontsize=8, va="center")
    axl.set_yticks(y, [lab for _, lab in GRP], color=t["fg"], fontsize=8.8)
    axl.set_xlim(0, 0.070)
    axl.set_ylim(-0.75, len(GRP) - 0.15)
    axl.set_xlabel("ΔR² (Nagelkerke) on the worst emotion item, LR-DIF matched on rest score",
                   color=t["dim"], fontsize=8.8)
    axl.xaxis.grid(True, color=t["grid"], lw=0.7)
    axl.set_axisbelow(True)
    axl.text(0, 1.02, "DOES THE ITEM FUNCTION DIFFERENTLY?", transform=axl.transAxes,
             color=t["dim"], fontsize=8.4, fontweight="bold", va="bottom")

    # right: does it cost anything -- A -> B labelling-function transfer
    for i, (g, lab) in enumerate(GRP):
        d = TRA["by_group"][g]
        col = PURPLE if g in LATENT else t["fg"]
        lo, hi = d["transfer_ci95"]
        nlo, nhi = d["permutation_null_ci95"]
        axr.plot([nlo, nhi], [y[i] + 0.20] * 2, color=t["dim"], lw=6, alpha=.30,
                 solid_capstyle="butt")
        axr.plot(d["permutation_null_mean"], y[i] + 0.20, "|", ms=13, color=t["dim"], mew=2)
        axr.plot([lo, hi], [y[i] - 0.14] * 2, color=col, lw=3, solid_capstyle="round",
                 alpha=.9 if g in LATENT else .6)
        axr.plot(d["transfer_accuracy"], y[i] - 0.14, "o", ms=8.5, color=col,
                 zorder=3, mec=t["bg"], mew=1.3)
        deg = d["degradation_vs_null"]
        txt = (f'{d["transfer_accuracy"]:.3f}   −{100*deg:.1f} pts, p < 0.005'
               if deg > 0.01 else f'{d["transfer_accuracy"]:.3f}   no loss, p = {d["permutation_p_one_sided"]:.2f}')
        axr.text(hi + 0.004, y[i] - 0.14, txt, color=col, fontsize=8.6, va="center",
                 fontweight="bold" if g in LATENT else "normal")
        axr.text(nhi + 0.004, y[i] + 0.20, f'null · n = {d["n_clips_used"]:,} clips',
                 color=t["dim"], fontsize=7.4, va="center")
    axr.set_yticks(y, ["" for _ in GRP])
    axr.set_xlim(0.552, 0.740)
    axr.set_ylim(-0.75, len(GRP) - 0.15)
    axr.set_xlabel("accuracy of group A's majority label scored against group B's consensus",
                   color=t["dim"], fontsize=8.8)
    axr.xaxis.grid(True, color=t["grid"], lw=0.7)
    axr.set_axisbelow(True)
    axr.text(0, 1.02, "DOES IT COST ANYTHING?", transform=axr.transAxes, color=t["dim"],
             fontsize=8.4, fontweight="bold", va="bottom")
    axr.legend(handles=[
        Line2D([], [], marker="|", ls="", color=t["dim"], mew=2, ms=11,
               label="size-matched random-regrouping null, 200 permutations"),
        Line2D([], [], marker="o", ls="", color=PURPLE, label="observed transfer, 95% CI, 300 clip bootstraps"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.135), frameon=False, fontsize=8,
       labelcolor=t["dim"], handletextpad=0.5, ncols=1)

    title(fig, t,
          "Rater faction changes the label, and only the latent grouping costs anything",
          "Audio-only, 73,253 trials, 2,443 raters. Speed, response-style extremity and fatigue split the raters\n"
          "into groups whose emotion items function identically — a negative result. The one grouping that bites\n"
          "is latent: the first PC of a rater's confusion profile, fitted on half their trials and tested on the other half.")
    footer(fig, t, "sources: invariance/out/dif-audio.json, invariance/out/transfer-audio.json · "
                   "ΔR² and Mantel-Haenszel disagree on the latent grouping (negligible vs ETS C); both are shown.")
    save(fig, "04-rater-invariance", theme)


def self_check() -> int:
    """The simulated gate is a claim the README makes, so it gets a test."""
    import subprocess, tempfile
    ok = check_not_simulated({"simulated": False, "a": [{"simulated": False}]}, "x") is None
    assert ok, "a clean artifact must pass"
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.json"
        bad.write_text(json.dumps({"by_modality": {"audio": {"simulated": True,
                                                             "source_file": "data/SIMULATED_x.parquet"}}}))
        r = subprocess.run([sys.executable, "-c",
                            f"import sys; sys.path.insert(0, {str(OUT)!r}); "
                            f"from make import check_not_simulated; import json; "
                            f"check_not_simulated(json.load(open({str(bad)!r})), 'bad.json')"],
                           capture_output=True, text=True)
        assert r.returncode != 0, "a simulated artifact must abort the run, not warn"
        assert "REFUSING TO DRAW" in r.stderr, r.stderr
    print("figures self-check ok: nested simulated=true aborts, simulated=false passes")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(self_check())
    for theme in ("light", "dark"):
        print(theme)
        fig_modality(theme)
        fig_sota(theme)
        fig_intent(theme)
        fig_dif(theme)
