# invariance/

Do the six emotion labels mean the same thing to every CREMA-D rater?

    .venv/bin/python invariance/nrm.py --check          # known-answer self-check
    .venv/bin/python invariance/nrm.py --mirt-check     # agree with R/mirt on a fixture
    .venv/bin/python invariance/nrm.py --modality audio --perm 200   # primary result
    .venv/bin/python invariance/dif.py --check
    .venv/bin/python invariance/dif.py --modality audio              # dichotomous DIF
    .venv/bin/python invariance/report.py                            # tables below

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

<!--REPORT-MIRT-->

## Primary result — audio only

Audio only, because that is the condition every CREMA-D speech-emotion benchmark scores
against and the condition where rater agreement is worst (Krippendorff's α = 0.265; see
`agreement/README.md`). Visual and audiovisual are reported as secondary below.

CREMA-D publishes no rater demographics, so the groupings are behavioural and are the
same ones the dichotomous analysis uses — see [`METHOD.md`](METHOD.md) for each one's
circularity risk. Every split is a median split, which costs power; that is inherited
deliberately so the NRM and the decile approximation are compared on identical groups.

<!--REPORT-AUDIO-->

<!--REPORT-COMPARE-->

## Leave-one-rater-out

<!--REPORT-LOO-->

## Secondary — visual and audiovisual

<!--REPORT-SECONDARY-->

## Robustness: keying on the crowd instead of the actor

<!--REPORT-CONSENSUS-->

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
   the failure mode directly: with one large planted effect, the five clean items pick up
   an apparent dTVD of up to 0.096 purely from anchor contamination, against a
   permutation null of 0.037. Read single-item results, not the profile across items.
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
