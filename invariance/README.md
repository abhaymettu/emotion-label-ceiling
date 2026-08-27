# invariance/

Do the six emotion labels mean the same thing to every CREMA-D rater?

**Not quite, and the amount depends on which emotion.** Under a fitted Bock nominal
response model on the audio-only condition (n = 73,253 trials, 2,443 raters, 7,442
clips), differential item functioning across behavioural rater groups is **small but
real**, and it is largest for `neutral` and `anger`. The biggest single effect is
`neutral` under a latent construal split: **10.3% of response mass moves between groups
at equal ability** (dTVD 0.1027, 95% CI 0.055–0.150, permutation null 0.0284,
p ≤ 0.005). Four of five groupings clear their null on at least one category; the
response-style-extremity split does not, on three of its six. Deleting any one of the
2,443 raters moves nothing by more than 0.002 on the manifest groupings.

The fitted model **agrees with the matched-decile stand-in it replaces on the ranking**
(Spearman ρ = +0.67) and **disagrees with it on magnitude** (the stand-in runs 1.93×
large in audio) **and on which wrong answer moves** (they name a different response
category in a third of cells). The last of those is the whole reason the nominal model
is the right tool.

    .venv/bin/python invariance/nrm.py --check          # known-answer self-check
    .venv/bin/python invariance/nrm.py --mirt-check     # agree with R/mirt on a fixture
    .venv/bin/python invariance/nrm.py --modality audio --perm 200   # primary result
    .venv/bin/python invariance/nrm.py --modality audio --loo --tag loo \
        --loo-groups grp_speed,grp_extremity,grp_position,grp_style,grp_style_resid
    .venv/bin/python invariance/nrm.py --modality audio --key consensus_loo --perm 200
    .venv/bin/python invariance/dif.py --check
    .venv/bin/python invariance/dif.py --modality audio              # dichotomous DIF
    .venv/bin/python invariance/dif.py --modality audio --key consensus_loo
    .venv/bin/python invariance/report.py --inject                   # tables below

Model justification and the design constraints that force it: [`METHOD.md`](METHOD.md).
Everything here is computed from `data/ratings_long.parquet` (219,686 real ratings).
No number on this page comes from `data/SIMULATED_*`; every script stamps
`"simulated": false` into its JSON and `report.py` refuses to print a table without it.

## What changed, and why it mattered

The repo's novelty claim is that prior annotator-psychometrics work is **ordinal** and
that a six-way **unordered nominal** forced choice needs different machinery. Until now
the analysis did not actually use that machinery: `nominal_shift` in `dif.py` compared
by-group response shares inside ability deciles and called it "Bock-style". That was a
stand-in for exactly the thing the claim is about.

It has been replaced by a fitted **Bock (1972) nominal response model**, and the decile
comparison is kept beside it, relabelled `decile_tvd_approx`, so the two can be
compared. They are compared below.

## The model

Items *i* are the six intended emotions. Categories *k* are the six response buttons.
Persons are raters. One latent rater trait θ.

$$P(\text{response}=k \mid \text{item } i, \theta) = \frac{\exp(a_{ik}\theta + c_{ik})}{\sum_h \exp(a_{ih}\theta + c_{ih})}$$

Nothing in that expression orders the six categories, which is the point. Fitted by
marginal maximum likelihood (Bock–Aitkin EM, 61 rectangle-rule quadrature nodes over
θ ∈ [−4, 4]). Identified by fixing $a_{i0}=c_{i0}=0$ (reference category `anger`) and
the reference group's θ at N(0,1); θ is oriented so the keyed category loads positively,
which makes higher θ mean "answers more like the actor was directed".

**DIF** is IRT-LR-DIF (Thissen, Steinberg & Wainer) adapted to nominal items. Baseline:
all six items constrained equal across the two rater groups, focal group's θ mean and SD
free. Alternative: the studied item's 2(K−1) = 10 parameters freed by group. Test
statistic 2ΔLL on 10 df.

**Effect size** is `dTVD`: the total-variation distance between the two groups' full
six-category response distributions **at equal θ**, averaged over the population θ
density. dTVD = 0.05 means that at a typical ability, five per cent of one group's
response mass sits on different buttons than the other group's. It is on the same scale
as the matched-decile TVD it replaces, deliberately, so the two are comparable.

### The p-value problem, and the permutation null

Freeing ten parameters always buys some dTVD, so dTVD has no zero point. And the χ²
reference distribution assumes trials are conditionally independent given θ, which this
design violates: a rater's five *fear* trials are five different clips, and clips differ
enormously in how recognisable they are. Measured directly
(`out/clip-heterogeneity-audio.json`, audio only):

| item | clips | mean keyed-response rate | observed clip-level variance | binomial-only variance | overdispersion |
|---|---|---|---|---|---|
| anger | 1,271 | 0.5155 | 0.08680 | 0.02532 | **3.43×** |
| fear | 1,271 | 0.3141 | 0.07527 | 0.02204 | **3.42×** |
| happy | 1,271 | 0.2832 | 0.06644 | 0.02078 | **3.20×** |
| disgust | 1,271 | 0.2829 | 0.04574 | 0.02076 | **2.20×** |
| sad | 1,271 | 0.2492 | 0.04166 | 0.01915 | **2.18×** |
| neutral | 1,087 | 0.7537 | 0.03185 | 0.01882 | **1.69×** |

Clip difficulty accounts for roughly **1.7–3.4× more variance than the model allows**
(median 2.7×). So the χ²(10) p-values printed below are anticonservative and are
reported as descriptive only.

The quantity to read instead is the **rater-permutation null**: relabel raters into
groups of the same sizes, refit, recompute dTVD, 200 replicates. That null inherits the
real clip structure and the real dependence, so `excess_dtvd_over_null` and `p(perm)`
are the calibrated numbers. For the within-rater grouping (session position) the
permutation swaps the early/late label *inside* each rater, which is the only
relabelling that preserves that design.

## Verification

Two independent checks, both runnable:

- `nrm.py --check` — generate data from a known NRM, recover its parameters; plant a
  nominal DIF effect that moves intercept mass from an item's keyed category onto
  `neutral` and confirm the machinery finds it, names `neutral` as the shifted category,
  gets the sign right, does not fire on the five clean items, and that the permutation
  null sits below the planted effect.
- `nrm.py --mirt-check` — fit the same NRM here and in R's `mirt` on a balanced fixture
  and compare fitted category response curves. (`mirt`'s default `itemtype="nominal"`
  is a *restricted* NRM — it fixes the top category's slope, one constraint per item
  more than identification needs — so `mirt_check.R` frees it first. Before freeing it,
  our log-likelihood beat mirt's by 14.05 on 6 fewer constraints, which is how the
  restriction was found.)

On a balanced fixture of 4,000 persons × 6 items × 6 categories: log-likelihood **-35821.509** here vs **-35821.62** in mirt (gap +0.111); largest disagreement between any two fitted category response curves anywhere on θ ∈ [−4, 4] is **0.0133** in probability (**0.0064** over |θ| ≤ 2.5), density-weighted mean **3.21e-04**. Two independent implementations, same answer.

## Primary result — audio only

Audio only, because that is the condition every CREMA-D speech-emotion benchmark scores
against and the condition where rater agreement is worst (Krippendorff's α = 0.265; see
`agreement/README.md`). Visual and audiovisual are reported as secondary below.

CREMA-D publishes no rater demographics, so the groupings are behavioural and are the
same ones the dichotomous analysis uses — see [`METHOD.md`](METHOD.md) for each one's
circularity risk. Every split is a median split, which costs power; that is inherited
deliberately so the NRM and the decile approximation are compared on identical groups.

### What it says

Read the `excess` and `p(perm)` columns, not `dTVD` or `p(χ²)`. `p(perm)` is floored at
1/201 = 0.005 by the number of permutations.

**DIF is small, real, and not uniform across categories.** Net of each item's own
permutation null, the ranking is **neutral +0.024, anger +0.020, fear +0.016, happy
+0.009, disgust +0.009, sad +0.008** averaged over the five groupings. The largest
single effect anywhere in the audio condition is *neutral* under the latent construal
split: dTVD 0.1027 against a null of 0.0284, excess **+0.074**, p(perm) = 0.005. Ten
points of response mass sit on different buttons between the two groups at equal θ.

**The negatives are real negatives and stay in.** Slider extremity sits below its
null on three of its six items (disgust −0.009, p = 0.97; happy −0.004; sad −0.003).
Response speed is below null on *sad* and *neutral*. Six of the thirty audio cells have a
negative excess, which is roughly what a genuinely null effect looks like.

**This is where the fitted model departs from the old headline.** The dichotomous
LR-DIF called every manifest grouping negligible on Jodoin–Gierl (largest ΔR² = 0.006),
and it is not wrong — but the nominal model finds response mass moving on the same
cells: response speed × *anger* excess **+0.040** (p = 0.005), session position ×
*anger* **+0.034** (p = 0.005), session position × *disgust* **+0.019** (p = 0.005),
slider extremity × *neutral* **+0.017** (p = 0.005). Both statements are true of the
same data and they answer different questions. ΔR² is trial-level variance explained,
which is tiny for any realistic effect on a six-way choice. dTVD is response mass moved.
A grouping can be negligible by the first and material by the second, and on this data
it is. **"Fatigue and response speed do not affect the labels" is not supported by the
nominal model; "they explain almost no trial-level variance" is.**

Source `data/ratings_long.parquet`, audio only: **n = 73,253 trials, 2,443 raters, 7,442 clips.**

| grouping | item | n trials | dTVD | perm. null | excess | p(perm) | χ²(10) | p(χ²) | largest shift |
|---|---|---|---|---|---|---|---|---|---|
| response speed | anger | 12,584 | 0.0624 | 0.0221 | +0.0402 | 0.005 | 64.3 | 5.61e-10 | `anger` +0.057 |
| response speed | disgust | 12,475 | 0.0404 | 0.0223 | +0.0181 | 0.010 | 24.1 | 0.00745 | `neutral` +0.027 |
| response speed | fear | 12,483 | 0.0343 | 0.0224 | +0.0119 | 0.025 | 32.1 | 0.000385 | `neutral` -0.015 |
| response speed | happy | 12,475 | 0.0282 | 0.0231 | +0.0051 | 0.209 | 22.8 | 0.0117 | `happy` +0.020 |
| response speed | sad | 12,474 | 0.0196 | 0.0202 | -0.0006 | 0.532 | 7.8 | 0.646 | `neutral` +0.015 |
| response speed | neutral | 10,762 | 0.0151 | 0.0201 | -0.0050 | 0.816 | 7.0 | 0.723 | `neutral` +0.011 |
| slider extremity | neutral | 10,762 | 0.0374 | 0.0200 | +0.0173 | 0.005 | 31.5 | 0.000479 | `neutral` -0.016 |
| slider extremity | anger | 12,584 | 0.0320 | 0.0222 | +0.0097 | 0.090 | 20.1 | 0.0285 | `disgust` -0.014 |
| slider extremity | fear | 12,483 | 0.0284 | 0.0223 | +0.0061 | 0.144 | 18.4 | 0.0492 | `neutral` +0.011 |
| slider extremity | happy | 12,475 | 0.0192 | 0.0231 | -0.0039 | 0.706 | 8.8 | 0.55 | `fear` -0.010 |
| slider extremity | sad | 12,474 | 0.0178 | 0.0205 | -0.0027 | 0.627 | 8.2 | 0.607 | `neutral` -0.012 |
| slider extremity | disgust | 12,475 | 0.0138 | 0.0230 | -0.0092 | 0.970 | 4.4 | 0.93 | `sad` +0.006 |
| session position *(within-rater)* | anger | 12,584 | 0.0542 | 0.0206 | +0.0336 | 0.005 | 42.7 | 5.59e-06 | `anger` +0.046 |
| session position *(within-rater)* | disgust | 12,475 | 0.0409 | 0.0223 | +0.0186 | 0.005 | 24.3 | 0.00695 | `neutral` -0.012 |
| session position *(within-rater)* | fear | 12,483 | 0.0337 | 0.0222 | +0.0114 | 0.040 | 16.7 | 0.0809 | `neutral` +0.017 |
| session position *(within-rater)* | sad | 12,474 | 0.0313 | 0.0192 | +0.0121 | 0.020 | 26.0 | 0.00376 | `neutral` +0.015 |
| session position *(within-rater)* | happy | 12,475 | 0.0239 | 0.0217 | +0.0021 | 0.333 | 11.1 | 0.347 | `neutral` -0.014 |
| session position *(within-rater)* | neutral | 10,762 | 0.0231 | 0.0193 | +0.0038 | 0.199 | 8.9 | 0.541 | `neutral` +0.021 |
| construal class (PC1) | neutral | 5,376 | 0.1027 | 0.0284 | +0.0743 | 0.005 | 35.6 | 9.76e-05 | `neutral` -0.094 |
| construal class (PC1) | fear | 6,202 | 0.0670 | 0.0318 | +0.0352 | 0.005 | 20.8 | 0.0223 | `fear` -0.040 |
| construal class (PC1) | happy | 6,187 | 0.0575 | 0.0322 | +0.0253 | 0.005 | 13.4 | 0.203 | `neutral` +0.028 |
| construal class (PC1) | sad | 6,222 | 0.0464 | 0.0298 | +0.0166 | 0.040 | 14.6 | 0.146 | `sad` -0.035 |
| construal class (PC1) | disgust | 6,320 | 0.0451 | 0.0326 | +0.0125 | 0.070 | 9.9 | 0.448 | `neutral` +0.032 |
| construal class (PC1) | anger | 6,303 | 0.0267 | 0.0318 | -0.0051 | 0.667 | 9.0 | 0.53 | `disgust` -0.011 |
| construal class, ability-residualised | neutral | 5,376 | 0.0582 | 0.0290 | +0.0291 | 0.005 | 19.2 | 0.0383 | `neutral` -0.037 |
| construal class, ability-residualised | anger | 6,303 | 0.0547 | 0.0313 | +0.0234 | 0.040 | 12.1 | 0.276 | `disgust` -0.027 |
| construal class, ability-residualised | fear | 6,202 | 0.0464 | 0.0319 | +0.0145 | 0.070 | 21.2 | 0.0196 | `happy` +0.011 |
| construal class, ability-residualised | sad | 6,222 | 0.0463 | 0.0297 | +0.0166 | 0.025 | 12.0 | 0.283 | `neutral` +0.038 |
| construal class, ability-residualised | happy | 6,187 | 0.0460 | 0.0312 | +0.0148 | 0.060 | 7.0 | 0.728 | `happy` +0.019 |
| construal class, ability-residualised | disgust | 6,320 | 0.0362 | 0.0330 | +0.0032 | 0.323 | 15.9 | 0.102 | `neutral` +0.016 |

Least invariant items, mean dTVD over the groupings: **neutral** 0.0473, **anger** 0.0460, **fear** 0.0420, **disgust** 0.0353, **happy** 0.0350, **sad** 0.0323.

Same ranking **net of each item's own permutation null**, which is the one to read because dTVD's floor scales with that item's n: **neutral** +0.0239, **anger** +0.0204, **fear** +0.0159, **happy** +0.0087, **disgust** +0.0086, **sad** +0.0084.

## Does the fitted model agree with the decile stand-in?

Across all **n = 30** (grouping × item) cells in this modality:

| compared | Pearson r | Spearman ρ |
|---|---|---|
| NRM dTVD vs matched-decile TVD | +0.529 | +0.667 |
| NRM dTVD above its permutation null vs decile TVD | +0.340 | +0.488 |
| NRM dTVD vs dichotomous LR-DIF ΔR² | +0.575 | +0.659 |
| NRM dTVD above null vs LR-DIF ΔR² | +0.490 | +0.529 |

Both methods name the same response category as the largest shift in **67%** of the 30 cells (chance with six categories is 17%).

The decile approximation is larger: median ratio decile TVD / NRM dTVD = **1.93**. It also concentrates on the modal response — it names `neutral` as the shifted category in **83%** of cells against the NRM's **60%**.

| grouping | item | NRM dTVD | decile TVD | NRM says | decile says |
|---|---|---|---|---|---|
| construal class (PC1) | neutral | 0.1027 | 0.0989 | `neutral` | `neutral` |
| construal class (PC1) | fear | 0.0670 | 0.1593 | `fear` | `neutral` |
| response speed | anger | 0.0624 | 0.0562 | `anger` | `anger` |
| construal class, ability-residualised | neutral | 0.0582 | 0.1289 | `neutral` | `neutral` |
| construal class (PC1) | happy | 0.0575 | 0.1541 | `neutral` | `neutral` |
| construal class, ability-residualised | anger | 0.0547 | 0.0858 | `disgust` | `neutral` |
| session position *(within-rater)* | anger | 0.0542 | 0.0727 | `anger` | `anger` |
| construal class (PC1) | sad | 0.0464 | 0.1893 | `sad` | `neutral` |

### What the comparison says

**The approximation gets the ordering roughly right and the magnitude and the
attribution wrong.** Rank correlation with the fitted model is +0.67 in audio and +0.78
in visual, so a reader ranking cells by the decile number would have got a broadly
similar list. But in audio the decile TVD is **1.93× larger** than the fitted dTVD at the
median (in visual it is 0.90×, so the bias is not even a constant), and it names
`neutral` as the shifted response in **83%** of cells against the fitted model's 60%. It
is drawn to the modal response, which in audio-only CREMA-D is overwhelmingly `neutral`
(75% of raters say `neutral` to a neutral clip and 38–53% say it to disgust, fear, happy
and sad clips — see `agreement/README.md`).

So: the stand-in agreed on *whether*, disagreed on *how much*, and disagreed on *which
wrong answer* in a third of cells. "Which wrong answer" is the entire reason the nominal
model is the right tool, so the disagreement lands exactly where it matters most.

## Leave-one-rater-out

Previously listed as pending and never run. Run now, exhaustively.

Every one of the **2,443** raters deleted in turn, the whole analysis refit each time (baseline plus six single-item models, 17,101 model fits per grouping).

| grouping | item | observed dTVD | 95% CI (jackknife) | min over deletions | max | largest single-rater move |
|---|---|---|---|---|---|---|
| response speed | anger | 0.0624 | 0.0423 – 0.0825 | 0.0616 | 0.0639 | 0.00147 |
| response speed | disgust | 0.0404 | 0.0222 – 0.0586 | 0.0397 | 0.0414 | 0.00096 |
| response speed | fear | 0.0343 | 0.0154 – 0.0533 | 0.0333 | 0.0351 | 0.00100 |
| response speed | happy | 0.0282 | 0.0129 – 0.0436 | 0.0269 | 0.0290 | 0.00134 |
| response speed | neutral | 0.0151 | -0.0028 – 0.0330 | 0.0144 | 0.0168 | 0.00175 |
| response speed | sad | 0.0196 | 0.0021 – 0.0371 | 0.0189 | 0.0211 | 0.00146 |
| slider extremity | anger | 0.0320 | 0.0124 – 0.0516 | 0.0305 | 0.0330 | 0.00148 |
| slider extremity | disgust | 0.0138 | -0.0041 – 0.0317 | 0.0128 | 0.0153 | 0.00145 |
| slider extremity | fear | 0.0284 | 0.0121 – 0.0448 | 0.0278 | 0.0296 | 0.00119 |
| slider extremity | happy | 0.0192 | 0.0020 – 0.0363 | 0.0186 | 0.0203 | 0.00116 |
| slider extremity | neutral | 0.0374 | 0.0204 – 0.0544 | 0.0362 | 0.0389 | 0.00153 |
| slider extremity | sad | 0.0178 | 0.0001 – 0.0355 | 0.0172 | 0.0190 | 0.00113 |
| session position *(within-rater)* | anger | 0.0542 | 0.0369 – 0.0715 | 0.0530 | 0.0553 | 0.00119 |
| session position *(within-rater)* | disgust | 0.0409 | 0.0220 – 0.0597 | 0.0405 | 0.0428 | 0.00197 |
| session position *(within-rater)* | fear | 0.0337 | 0.0153 – 0.0520 | 0.0331 | 0.0352 | 0.00158 |
| session position *(within-rater)* | happy | 0.0239 | 0.0062 – 0.0415 | 0.0234 | 0.0250 | 0.00110 |
| session position *(within-rater)* | neutral | 0.0231 | 0.0071 – 0.0390 | 0.0222 | 0.0239 | 0.00085 |
| session position *(within-rater)* | sad | 0.0313 | 0.0125 – 0.0500 | 0.0309 | 0.0329 | 0.00168 |
| construal class (PC1) | anger | 0.0267 | 0.0004 – 0.0529 | 0.0248 | 0.0296 | 0.00291 |
| construal class (PC1) | disgust | 0.0451 | 0.0125 – 0.0777 | 0.0419 | 0.0502 | 0.00514 |
| construal class (PC1) | fear | 0.0670 | 0.0355 – 0.0985 | 0.0650 | 0.0695 | 0.00249 |
| construal class (PC1) | happy | 0.0575 | 0.0273 – 0.0877 | 0.0573 | 0.0621 | 0.00460 |
| construal class (PC1) | neutral | 0.1027 | 0.0553 – 0.1501 | 0.0875 | 0.1067 | 0.01524 |
| construal class (PC1) | sad | 0.0464 | 0.0176 – 0.0752 | 0.0445 | 0.0490 | 0.00256 |
| construal class, ability-residualised | anger | 0.0547 | 0.0217 – 0.0877 | 0.0531 | 0.0592 | 0.00447 |
| construal class, ability-residualised | disgust | 0.0362 | 0.0078 – 0.0646 | 0.0339 | 0.0381 | 0.00235 |
| construal class, ability-residualised | fear | 0.0464 | 0.0164 – 0.0764 | 0.0447 | 0.0479 | 0.00170 |
| construal class, ability-residualised | happy | 0.0460 | 0.0104 – 0.0817 | 0.0447 | 0.0501 | 0.00402 |
| construal class, ability-residualised | neutral | 0.0582 | 0.0231 – 0.0932 | 0.0526 | 0.0621 | 0.00560 |
| construal class, ability-residualised | sad | 0.0463 | 0.0157 – 0.0768 | 0.0442 | 0.0500 | 0.00372 |

The CI is the delete-one jackknife, ±1.96 SE. dTVD contains an absolute value, so it is not everywhere smooth and the jackknife is approximate near dTVD ≈ 0; read it as a scale, not an exact interval. It is an interval on dTVD itself, not on dTVD net of its permutation null.

**Nothing rests on one rater.** On the three manifest groupings the largest move any
single deletion produces is **0.0020** in dTVD, against effects of 0.014–0.062. On the
latent construal split the deletions bite harder — up to **0.0152** on the headline
`neutral` cell, about 15% of its value — but even the minimum over all 2,443 deletions
(0.0875) stays far above that cell's permutation null of 0.0284. The construal groupings
are the ones to treat as least stable, which is consistent with their being estimated on
half the trials (the PCA holdout) and defined from the responses themselves.

One methodological note, because it nearly became a false finding: the first LOO run
warm-started every refit from the *anchored* baseline with a loose tolerance, which drags
dTVD toward zero and put the observed value **outside** the LOO range in several cells —
which reads exactly like "deleting any rater lowers the estimate". It was numerical, not
empirical. Refits now start from their own full-sample solution and `nrm.py --check`
asserts that LOO brackets the full-sample estimate.

## Secondary — visual and audiovisual

### visual only

Source `data/ratings_long.parquet`, visual only: **n = 73,191 trials, 2,443 raters, 7,442 clips.**

| grouping | item | n trials | dTVD | perm. null | excess | p(perm) | χ²(10) | p(χ²) | largest shift |
|---|---|---|---|---|---|---|---|---|---|
| response speed | happy | 12,540 | 0.0517 | 0.0293 | +0.0224 | 0.100 | 48.8 | 4.35e-07 | `happy` +0.036 |
| response speed | anger | 12,497 | 0.0458 | 0.0242 | +0.0216 | 0.005 | 40.4 | 1.47e-05 | `disgust` -0.037 |
| response speed | fear | 12,526 | 0.0420 | 0.0266 | +0.0154 | 0.020 | 28.7 | 0.00138 | `fear` +0.021 |
| response speed | neutral | 10,703 | 0.0401 | 0.0226 | +0.0175 | 0.010 | 28.4 | 0.00158 | `neutral` +0.009 |
| response speed | disgust | 12,468 | 0.0367 | 0.0257 | +0.0110 | 0.109 | 23.0 | 0.0106 | `disgust` +0.027 |
| response speed | sad | 12,457 | 0.0350 | 0.0229 | +0.0121 | 0.060 | 19.2 | 0.0384 | `sad` -0.029 |
| slider extremity | happy | 12,540 | 0.0429 | 0.0270 | +0.0159 | 0.114 | 34.1 | 0.000174 | `disgust` +0.003 |
| slider extremity | disgust | 12,468 | 0.0309 | 0.0253 | +0.0056 | 0.184 | 13.2 | 0.212 | `anger` +0.009 |
| slider extremity | anger | 12,497 | 0.0222 | 0.0226 | -0.0003 | 0.468 | 11.3 | 0.336 | `anger` +0.007 |
| slider extremity | sad | 12,457 | 0.0214 | 0.0229 | -0.0015 | 0.572 | 11.0 | 0.356 | `disgust` +0.015 |
| slider extremity | neutral | 10,703 | 0.0189 | 0.0232 | -0.0043 | 0.721 | 8.8 | 0.555 | `neutral` -0.012 |
| slider extremity | fear | 12,526 | 0.0181 | 0.0256 | -0.0075 | 0.866 | 10.6 | 0.392 | `sad` -0.005 |
| session position *(within-rater)* | fear | 12,526 | 0.0460 | 0.0239 | +0.0220 | 0.005 | 34.6 | 0.000147 | `fear` -0.023 |
| session position *(within-rater)* | anger | 12,497 | 0.0409 | 0.0224 | +0.0185 | 0.015 | 28.4 | 0.00157 | `anger` +0.029 |
| session position *(within-rater)* | disgust | 12,468 | 0.0283 | 0.0235 | +0.0048 | 0.219 | 20.0 | 0.0292 | `anger` +0.011 |
| session position *(within-rater)* | sad | 12,457 | 0.0264 | 0.0227 | +0.0037 | 0.284 | 13.0 | 0.223 | `neutral` +0.017 |
| session position *(within-rater)* | happy | 12,540 | 0.0241 | 0.0182 | +0.0059 | 0.239 | 23.5 | 0.009 | `neutral` -0.008 |
| session position *(within-rater)* | neutral | 10,703 | 0.0226 | 0.0226 | +0.0000 | 0.463 | 9.9 | 0.453 | `neutral` +0.013 |
| construal class (PC1) | anger | 6,193 | 0.0685 | 0.0317 | +0.0368 | 0.005 | 35.4 | 0.000105 | `anger` -0.062 |
| construal class (PC1) | disgust | 6,311 | 0.0641 | 0.0355 | +0.0287 | 0.005 | 30.9 | 0.000605 | `disgust` +0.041 |
| construal class (PC1) | neutral | 5,249 | 0.0637 | 0.0321 | +0.0316 | 0.005 | 19.0 | 0.0407 | `neutral` -0.041 |
| construal class (PC1) | sad | 6,239 | 0.0511 | 0.0329 | +0.0182 | 0.040 | 18.7 | 0.0437 | `neutral` +0.025 |
| construal class (PC1) | fear | 6,314 | 0.0495 | 0.0365 | +0.0130 | 0.129 | 17.4 | 0.0664 | `disgust` +0.024 |
| construal class (PC1) | happy | 6,272 | 0.0436 | 0.0370 | +0.0066 | 0.289 | 31.3 | 0.000521 | `happy` -0.033 |
| construal class, ability-residualised | anger | 6,193 | 0.0672 | 0.0317 | +0.0356 | 0.005 | 34.9 | 0.000128 | `anger` -0.060 |
| construal class, ability-residualised | neutral | 5,249 | 0.0633 | 0.0320 | +0.0314 | 0.010 | 19.3 | 0.0364 | `neutral` -0.040 |
| construal class, ability-residualised | disgust | 6,311 | 0.0625 | 0.0354 | +0.0271 | 0.005 | 30.2 | 0.000791 | `disgust` +0.038 |
| construal class, ability-residualised | fear | 6,314 | 0.0494 | 0.0366 | +0.0128 | 0.124 | 17.4 | 0.0667 | `disgust` +0.024 |
| construal class, ability-residualised | sad | 6,239 | 0.0480 | 0.0331 | +0.0149 | 0.055 | 16.4 | 0.0895 | `neutral` +0.023 |
| construal class, ability-residualised | happy | 6,272 | 0.0454 | 0.0372 | +0.0083 | 0.259 | 32.3 | 0.000352 | `happy` -0.033 |

Least invariant items, mean dTVD over the groupings: **anger** 0.0489, **disgust** 0.0445, **neutral** 0.0417, **happy** 0.0415, **fear** 0.0410, **sad** 0.0364.

Same ranking **net of each item's own permutation null**, which is the one to read because dTVD's floor scales with that item's n: **anger** +0.0224, **disgust** +0.0154, **neutral** +0.0152, **happy** +0.0118, **fear** +0.0111, **sad** +0.0095.


Across all **n = 30** (grouping × item) cells in this modality:

| compared | Pearson r | Spearman ρ |
|---|---|---|
| NRM dTVD vs matched-decile TVD | +0.721 | +0.782 |
| NRM dTVD above its permutation null vs decile TVD | +0.640 | +0.644 |
| NRM dTVD vs dichotomous LR-DIF ΔR² | +0.430 | +0.611 |
| NRM dTVD above null vs LR-DIF ΔR² | +0.407 | +0.603 |

Both methods name the same response category as the largest shift in **57%** of the 30 cells (chance with six categories is 17%).

The decile approximation is larger: median ratio decile TVD / NRM dTVD = **0.90**. It also concentrates on the modal response — it names `neutral` as the shifted category in **57%** of cells against the NRM's **30%**.

| grouping | item | NRM dTVD | decile TVD | NRM says | decile says |
|---|---|---|---|---|---|
| construal class (PC1) | anger | 0.0685 | 0.0738 | `anger` | `neutral` |
| construal class, ability-residualised | anger | 0.0672 | 0.0734 | `anger` | `neutral` |
| construal class (PC1) | disgust | 0.0641 | 0.0427 | `disgust` | `disgust` |
| construal class (PC1) | neutral | 0.0637 | 0.1407 | `neutral` | `neutral` |
| construal class, ability-residualised | neutral | 0.0633 | 0.1392 | `neutral` | `neutral` |
| construal class, ability-residualised | disgust | 0.0625 | 0.0393 | `disgust` | `disgust` |
| response speed | happy | 0.0517 | 0.0157 | `happy` | `happy` |
| construal class (PC1) | sad | 0.0511 | 0.0615 | `neutral` | `neutral` |

### audiovisual

Source `data/ratings_long.parquet`, audiovisual: **n = 73,242 trials, 2,443 raters, 7,442 clips.**

| grouping | item | n trials | dTVD | perm. null | excess | p(perm) | χ²(10) | p(χ²) | largest shift |
|---|---|---|---|---|---|---|---|---|---|
| response speed | happy | 12,562 | 0.0660 | 0.0206 | +0.0454 | 0.005 | 89.0 | 8.63e-15 | `happy` +0.064 |
| response speed | disgust | 12,498 | 0.0580 | 0.0229 | +0.0352 | 0.005 | 52.3 | 9.89e-08 | `disgust` +0.045 |
| response speed | anger | 12,478 | 0.0580 | 0.0235 | +0.0345 | 0.005 | 63.0 | 9.78e-10 | `anger` +0.056 |
| response speed | neutral | 10,656 | 0.0546 | 0.0229 | +0.0318 | 0.005 | 57.2 | 1.21e-08 | `neutral` +0.036 |
| response speed | sad | 12,502 | 0.0409 | 0.0248 | +0.0161 | 0.025 | 28.2 | 0.0017 | `neutral` +0.025 |
| response speed | fear | 12,546 | 0.0320 | 0.0240 | +0.0079 | 0.095 | 32.5 | 0.000334 | `neutral` +0.012 |
| slider extremity | sad | 12,502 | 0.0478 | 0.0251 | +0.0227 | 0.010 | 26.0 | 0.00378 | `neutral` -0.020 |
| slider extremity | happy | 12,562 | 0.0302 | 0.0213 | +0.0088 | 0.144 | 25.4 | 0.0046 | `happy` -0.012 |
| slider extremity | fear | 12,546 | 0.0257 | 0.0251 | +0.0006 | 0.418 | 16.2 | 0.0938 | `fear` -0.008 |
| slider extremity | disgust | 12,498 | 0.0220 | 0.0227 | -0.0008 | 0.517 | 7.4 | 0.684 | `anger` +0.004 |
| slider extremity | anger | 12,478 | 0.0177 | 0.0242 | -0.0065 | 0.806 | 9.0 | 0.529 | `anger` +0.004 |
| slider extremity | neutral | 10,656 | 0.0172 | 0.0226 | -0.0054 | 0.746 | 11.4 | 0.329 | `fear` -0.006 |
| session position *(within-rater)* | neutral | 10,656 | 0.0347 | 0.0185 | +0.0163 | 0.010 | 25.2 | 0.00505 | `neutral` +0.026 |
| session position *(within-rater)* | anger | 12,478 | 0.0272 | 0.0210 | +0.0062 | 0.169 | 19.3 | 0.0367 | `anger` +0.018 |
| session position *(within-rater)* | fear | 12,546 | 0.0190 | 0.0210 | -0.0020 | 0.617 | 6.2 | 0.799 | `disgust` -0.004 |
| session position *(within-rater)* | sad | 12,502 | 0.0170 | 0.0221 | -0.0051 | 0.801 | 5.0 | 0.893 | `neutral` +0.002 |
| session position *(within-rater)* | happy | 12,562 | 0.0126 | 0.0163 | -0.0037 | 0.677 | 8.1 | 0.619 | `happy` +0.007 |
| session position *(within-rater)* | disgust | 12,498 | 0.0081 | 0.0215 | -0.0134 | 0.995 | 1.9 | 0.997 | `anger` +0.004 |
| construal class (PC1) | neutral | 5,280 | 0.1023 | 0.0322 | +0.0701 | 0.005 | 55.8 | 2.28e-08 | `neutral` -0.098 |
| construal class (PC1) | happy | 6,228 | 0.0889 | 0.0273 | +0.0616 | 0.005 | 63.1 | 9.4e-10 | `happy` -0.067 |
| construal class (PC1) | anger | 6,354 | 0.0762 | 0.0351 | +0.0412 | 0.020 | 37.9 | 4e-05 | `disgust` +0.062 |
| construal class (PC1) | fear | 6,281 | 0.0708 | 0.0344 | +0.0364 | 0.005 | 27.4 | 0.00223 | `fear` -0.034 |
| construal class (PC1) | disgust | 6,199 | 0.0520 | 0.0339 | +0.0181 | 0.030 | 22.2 | 0.014 | `disgust` +0.038 |
| construal class (PC1) | sad | 6,261 | 0.0495 | 0.0337 | +0.0158 | 0.045 | 34.4 | 0.000155 | `disgust` +0.024 |
| construal class, ability-residualised | neutral | 5,280 | 0.0908 | 0.0327 | +0.0581 | 0.005 | 48.6 | 4.75e-07 | `neutral` -0.084 |
| construal class, ability-residualised | happy | 6,228 | 0.0721 | 0.0273 | +0.0448 | 0.005 | 49.9 | 2.77e-07 | `happy` -0.059 |
| construal class, ability-residualised | fear | 6,281 | 0.0684 | 0.0343 | +0.0341 | 0.005 | 25.0 | 0.00527 | `fear` -0.029 |
| construal class, ability-residualised | anger | 6,354 | 0.0658 | 0.0354 | +0.0304 | 0.025 | 30.6 | 0.000686 | `disgust` +0.051 |
| construal class, ability-residualised | disgust | 6,199 | 0.0572 | 0.0337 | +0.0236 | 0.020 | 24.2 | 0.00713 | `disgust` +0.041 |
| construal class, ability-residualised | sad | 6,261 | 0.0442 | 0.0337 | +0.0105 | 0.119 | 30.0 | 0.000854 | `disgust` +0.022 |

Least invariant items, mean dTVD over the groupings: **neutral** 0.0599, **happy** 0.0539, **anger** 0.0490, **fear** 0.0432, **sad** 0.0399, **disgust** 0.0395.

Same ranking **net of each item's own permutation null**, which is the one to read because dTVD's floor scales with that item's n: **neutral** +0.0342, **happy** +0.0314, **anger** +0.0211, **fear** +0.0154, **disgust** +0.0125, **sad** +0.0120.


Across all **n = 30** (grouping × item) cells in this modality:

| compared | Pearson r | Spearman ρ |
|---|---|---|
| NRM dTVD vs matched-decile TVD | +0.702 | +0.705 |
| NRM dTVD above its permutation null vs decile TVD | +0.648 | +0.657 |
| NRM dTVD vs dichotomous LR-DIF ΔR² | +0.578 | +0.521 |
| NRM dTVD above null vs LR-DIF ΔR² | +0.596 | +0.551 |

Both methods name the same response category as the largest shift in **70%** of the 30 cells (chance with six categories is 17%).

The decile approximation is larger: median ratio decile TVD / NRM dTVD = **0.74**. It also concentrates on the modal response — it names `neutral` as the shifted category in **50%** of cells against the NRM's **27%**.

| grouping | item | NRM dTVD | decile TVD | NRM says | decile says |
|---|---|---|---|---|---|
| construal class (PC1) | neutral | 0.1023 | 0.1483 | `neutral` | `neutral` |
| construal class, ability-residualised | neutral | 0.0908 | 0.1434 | `neutral` | `neutral` |
| construal class (PC1) | happy | 0.0889 | 0.0146 | `happy` | `neutral` |
| construal class (PC1) | anger | 0.0762 | 0.0583 | `disgust` | `disgust` |
| construal class, ability-residualised | happy | 0.0721 | 0.0298 | `happy` | `neutral` |
| construal class (PC1) | fear | 0.0708 | 0.0419 | `fear` | `neutral` |
| construal class, ability-residualised | fear | 0.0684 | 0.0358 | `fear` | `neutral` |
| response speed | happy | 0.0660 | 0.0489 | `happy` | `happy` |

Same picture, different categories. In visual the least invariant item is **anger**
(+0.022 net of null), not neutral, and the construal split moves `anger` mass by
−0.062. In audiovisual response speed is the strongest grouping, with *happy* at
excess +0.045. Nothing here contradicts the audio result; it says the category that
functions least equivalently depends on which channel the rater is given, which is
what you would expect and is not evidence for anything on its own.

## Robustness: keying on the crowd instead of the actor

Items redefined by the **leave-one-rater-out crowd majority** label instead of the actor's direction — the rater's own vote is dropped before the majority is taken, so a rater is never scored against a label they helped build. Item sizes become very unbalanced under this key (audio consensus is dominated by `neutral`), so dTVD is not comparable across items here; read each item against its own permutation null.

Source `data/ratings_long.parquet`, audio only: **n = 73,253 trials, 2,443 raters, 7,442 clips.**

| grouping | item | n trials | dTVD | perm. null | excess | p(perm) | χ²(10) | p(χ²) | largest shift |
|---|---|---|---|---|---|---|---|---|---|
| response speed | anger | 10,391 | 0.0696 | 0.0235 | +0.0460 | 0.005 | 72.6 | 1.36e-11 | `anger` +0.056 |
| response speed | happy | 3,995 | 0.0570 | 0.0357 | +0.0213 | 0.035 | 23.0 | 0.0106 | `happy` +0.041 |
| response speed | disgust | 6,487 | 0.0399 | 0.0307 | +0.0092 | 0.114 | 14.9 | 0.137 | `disgust` -0.032 |
| response speed | fear | 7,119 | 0.0373 | 0.0287 | +0.0086 | 0.124 | 28.5 | 0.00148 | `fear` -0.020 |
| response speed | sad | 4,584 | 0.0357 | 0.0342 | +0.0015 | 0.393 | 14.5 | 0.15 | `fear` -0.023 |
| response speed | neutral | 40,677 | 0.0252 | 0.0136 | +0.0117 | 0.015 | 20.9 | 0.0217 | `neutral` +0.025 |
| slider extremity | disgust | 6,487 | 0.0414 | 0.0292 | +0.0122 | 0.075 | 19.4 | 0.0351 | `disgust` -0.029 |
| slider extremity | happy | 3,995 | 0.0375 | 0.0374 | +0.0002 | 0.463 | 15.4 | 0.119 | `fear` -0.025 |
| slider extremity | sad | 4,584 | 0.0358 | 0.0346 | +0.0011 | 0.458 | 13.7 | 0.187 | `neutral` +0.017 |
| slider extremity | fear | 7,119 | 0.0297 | 0.0286 | +0.0011 | 0.468 | 9.3 | 0.507 | `neutral` +0.008 |
| slider extremity | neutral | 40,677 | 0.0214 | 0.0132 | +0.0082 | 0.025 | 24.8 | 0.00568 | `neutral` -0.012 |
| slider extremity | anger | 10,391 | 0.0158 | 0.0224 | -0.0066 | 0.841 | 13.9 | 0.179 | `anger` -0.006 |
| session position *(within-rater)* | anger | 10,391 | 0.0662 | 0.0215 | +0.0447 | 0.005 | 51.9 | 1.17e-07 | `anger` +0.053 |
| session position *(within-rater)* | sad | 4,584 | 0.0631 | 0.0329 | +0.0302 | 0.010 | 22.2 | 0.0142 | `neutral` +0.020 |
| session position *(within-rater)* | disgust | 6,487 | 0.0503 | 0.0296 | +0.0207 | 0.010 | 22.6 | 0.0124 | `neutral` -0.038 |
| session position *(within-rater)* | happy | 3,995 | 0.0500 | 0.0345 | +0.0155 | 0.060 | 13.1 | 0.221 | `happy` +0.030 |
| session position *(within-rater)* | fear | 7,119 | 0.0444 | 0.0276 | +0.0168 | 0.020 | 21.6 | 0.0172 | `neutral` +0.016 |
| session position *(within-rater)* | neutral | 40,677 | 0.0353 | 0.0116 | +0.0237 | 0.005 | 42.0 | 7.39e-06 | `neutral` +0.031 |
| construal class (PC1) | sad | 2,323 | 0.1174 | 0.0486 | +0.0687 | 0.005 | 22.7 | 0.0119 | `sad` -0.071 |
| construal class (PC1) | happy | 1,968 | 0.0867 | 0.0528 | +0.0340 | 0.020 | 13.4 | 0.201 | `fear` +0.020 |
| construal class (PC1) | fear | 3,559 | 0.0684 | 0.0419 | +0.0266 | 0.005 | 14.5 | 0.151 | `sad` +0.029 |
| construal class (PC1) | anger | 5,152 | 0.0522 | 0.0339 | +0.0183 | 0.035 | 11.8 | 0.297 | `anger` -0.040 |
| construal class (PC1) | disgust | 3,273 | 0.0455 | 0.0448 | +0.0007 | 0.448 | 5.1 | 0.882 | `neutral` +0.036 |
| construal class (PC1) | neutral | 20,335 | 0.0173 | 0.0179 | -0.0006 | 0.512 | 9.4 | 0.497 | `happy` +0.010 |
| construal class, ability-residualised | sad | 2,323 | 0.0876 | 0.0474 | +0.0402 | 0.015 | 19.9 | 0.0304 | `sad` -0.046 |
| construal class, ability-residualised | happy | 1,968 | 0.0827 | 0.0515 | +0.0313 | 0.015 | 13.0 | 0.221 | `happy` +0.045 |
| construal class, ability-residualised | fear | 3,559 | 0.0558 | 0.0406 | +0.0151 | 0.100 | 14.1 | 0.168 | `fear` +0.014 |
| construal class, ability-residualised | anger | 5,152 | 0.0506 | 0.0331 | +0.0176 | 0.055 | 11.1 | 0.346 | `disgust` -0.020 |
| construal class, ability-residualised | neutral | 20,335 | 0.0438 | 0.0175 | +0.0263 | 0.005 | 19.4 | 0.0357 | `neutral` +0.041 |
| construal class, ability-residualised | disgust | 3,273 | 0.0234 | 0.0450 | -0.0216 | 0.975 | 1.6 | 0.999 | `neutral` +0.019 |

Least invariant items, mean dTVD over the groupings: **sad** 0.0679, **happy** 0.0628, **anger** 0.0509, **fear** 0.0471, **disgust** 0.0401, **neutral** 0.0286.

Same ranking **net of each item's own permutation null**, which is the one to read because dTVD's floor scales with that item's n: **sad** +0.0284, **anger** +0.0240, **happy** +0.0204, **neutral** +0.0138, **fear** +0.0136, **disgust** +0.0043.

## Assumptions

Stated whether or not they help.

1. **Unidimensionality.** One latent rater trait explains all six items. Six items is
   too few to test dimensionality usefully, so this is assumed and not checked. If rater
   ability is genuinely multidimensional — say, a voice-sensitivity factor separate from
   a response-bias factor — matching on a single θ is the wrong conditioning and the DIF
   estimates are biased in an unknown direction.
2. **Conditional independence of trials given θ.** Violated, measurably, by clip
   difficulty (1.7–3.4× overdispersion, table above). This is why the permutation null
   exists and why the χ² is labelled descriptive.
3. **Anchor purity.** Testing item *i* assumes the other five function identically
   across groups. Standard practice, and wrong if DIF is pervasive. The self-check shows
   the failure mode directly: with one large effect planted on `fear`, the five clean
   items pick up an apparent dTVD of up to 0.096 purely from anchor contamination —
   against 0.037, the largest dTVD in twelve replicates of the same data with the group
   labels scrambled. Read single-item results, not the profile across items.
4. **Normal θ within group.** Assumed. The focal group's mean and SD are free; its shape
   is not.
5. **Median splits of continuous covariates.** Speed, extremity and construal style are
   continuous and are dichotomised. This throws away information and biases every effect
   size toward zero.
6. **Items are emotion categories, not tasks.** A rater's five *fear* trials are five
   different clips, so an "item" here is a category of stimulus, not a fixed question.
   This is forced by the design: ~10 raters per clip cannot support per-clip DIF at 7,442
   items.
7. **The key is the actor's direction.** Correctness, and θ's orientation, are scored
   against what the actor was *told to portray*, not against truth. This buys a criterion
   external to every rater, which is what stops the analysis being circular; it costs the
   assumption that direction and signal coincide. The consensus-keyed run above is the
   check on that, and the NRM's DIF test itself does not use a key at all — it models the
   whole response distribution, so only θ's orientation and the `keyed_diff` column
   depend on the key.
8. **Two groups, behavioural.** No rater demographics exist in CREMA-D. Nothing here
   says anything about DIF by rater age, sex, culture, or language background, which are
   the groupings anyone would actually want.
9. **Session position splits one human into two person units** with independent θ. They
   are the same person, so their θ are correlated and the model ignores that, which
   understates uncertainty on that grouping specifically.

## What this design cannot support

- **Per-clip DIF.** ~10 raters per clip per modality. Not estimable.
- **Demographic DIF.** No rater demographics in the corpus. Not estimable, at any n.
- **Any causal statement about why groups differ.** The groupings are behavioural
  summaries of the same responses being modelled; `grp_style` is partly circular by
  construction and is decontaminated by a train/test split, not by design.
- **Generalisation past acted emotion.** CREMA-D actors were directed. Spontaneous
  affect has no such anchor and there is no reason to expect these magnitudes to carry.
- **Generalisation past this rater pool.** 2,443 crowd workers, one platform, one
  recruitment window.
- **A claim that the crowd label is wrong.** DIF says the items function differently
  across rater groups. Whether that costs anything is the separate question
  `transfer.py` answers.

## Files

| Path | What |
|---|---|
| `nrm.py` | Bock NRM, multigroup MML-EM, IRT-LR-DIF, permutation null, leave-one-rater-out, self-check, mirt cross-check |
| `dif.py` | dichotomous LR-DIF (Swaminathan & Rogers) + Mantel–Haenszel + the matched-decile approximation, self-check |
| `transfer.py` | A→B labelling-function transfer with a size-matched permutation null |
| `mirt_check.R` | independent NRM fit in R/`mirt` for the cross-check |
| `report.py` | regenerates every table on this page from `out/*.json` |
| `METHOD.md` | why these models and not others |
| `out/nrm-dif-{modality}.{csv,json}` | NRM DIF results |
| `out/nrm-dif-audio-loo.json` | leave-one-rater-out refits |
| `out/dif-{modality}.{csv,json}` | dichotomous DIF results, incl. `decile_tvd_approx` |
| `out/clip-heterogeneity-audio.json` | the overdispersion table above |

`out/` is gitignored and regenerable; delete and re-run is always safe.
