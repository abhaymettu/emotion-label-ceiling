# Measurement invariance across annotators

## The question

A benchmark treats "anger" as one thing. Every rater's `anger` response is pooled
into one consensus label as though the six response options mean the same thing to
everyone. That is an **invariance assumption**, and in psychometrics it is a
hypothesis you test, not a convention you adopt.

Formally, an item exhibits **differential item functioning** if two respondents at
the same level of the latent trait have different probabilities of a given response,
conditional on group. Adapted here:

> Two raters who are equally good at recognising acted emotion overall should have
> the same probability of labelling a *fear* clip `fear`. If rater group membership
> still predicts that probability after conditioning on overall ability, the item
> does not function equivalently and the groups are not measuring the same thing.

## The design, and what it forces

Every rater in CREMA-D saw all three presentation modalities (2,442 of 2,443 saw
all three), roughly 90 clips each, 30 per modality, 5 per emotion per modality.
Clips are crossed with raters only sparsely — a given clip gets ~10 raters per
modality out of 2,443. So:

- **Raters are persons. The six emotion categories are items.** A rater's score on
  item *c* is the proportion of intended-*c* clips they labelled *c*. Six items,
  2,443 persons, ~15 trials per person-item pooled over modality. Dense enough
  to fit. Clips-as-items is not viable — ~10 raters per clip cannot support
  per-clip DIF at 7,442 items.
- Modality is **within-rater and balanced**, so it does not confound between-rater
  group comparisons. It is carried as a covariate and analysed separately.

## Why this model and not another

| Candidate | Verdict |
|---|---|
| **Multi-group CFA, configural → metric → scalar** | Rejected. Scalar invariance testing needs continuous or ordered-categorical indicators with a factor structure and estimable intercepts/thresholds. A six-way **unordered** forced choice has neither. Fitting one anyway would require pretending the response is ordinal, which it is not. |
| **Logistic-regression DIF (Swaminathan & Rogers, 1990)** | **Primary.** Dichotomous item scores, an explicit matching variable, separate uniform and non-uniform terms, and a calibrated effect size. Transparent about what is conditioned on. |
| **Mantel–Haenszel with ETS A/B/C classification** | **Cross-check.** No parametric form at all, and the classification thresholds are the ones an operational testing programme would use. Detects only uniform DIF, which is why it is secondary. |
| **Nominal Response Model (Bock, 1972) style extension** | **Secondary, and the interesting one.** Dichotomising to correct/incorrect throws away *which* wrong answer was given. A multinomial logit over the six response options, matched on ability, with a group term, recovers that. "Group A hears fear as sadness, group B hears it as fear" is invisible to LR-DIF and visible here. |
| **2PL/3PL IRT with Lord's chi-square** | Rejected as primary. Adds parametric assumptions (a latent normal ability, local independence across items) for no gain over LR-DIF at six items, and local independence is doubtful when the six items are scored from one interleaved session. |

## The matching variable

**Rest score**, not total score: a rater's ability on item *c* is matched on their
accuracy across the *other five* items. Matching on the total score lets the studied
item contaminate its own matching criterion and biases DIF toward zero. This is a
standard correction and it is frequently skipped.

## The keyed response

Correctness is scored against the **intended** (actor-directed) emotion, not the
crowd consensus. The consensus label is built out of the very raters whose
functioning is under test; using it as the key would make the analysis circular by
construction. The intended label is external to every rater. A leave-one-rater-out
consensus key is reported as a robustness check.

This has a cost, stated plainly: intended emotion is what the actor was told to
portray, not necessarily what is in the signal. Scoring against it measures
agreement with the director, not truth.

## Grouping variables

CREMA-D publishes no rater demographics. The available covariates are behavioural,
and each has a different circularity risk:

| Grouping | Definition | Circular? |
|---|---|---|
| `speed` | Median split on the rater's own median response time | **No.** Timing is independent of which label was chosen. |
| `extremity` | Median split on the rater's mean \|intensity − 50\|, i.e. extreme vs moderate response style | **No.** Uses the intensity slider, not the emotion choice. Extreme responding is a classic psychometric response style. |
| `position` | Within-rater: first half vs second half of the session (`question_num`) | **No**, and it is a *within-person* invariance test — the same rater against themselves over time. |
| `style` | Sign of the first principal component of the rater's 6×6 confusion profile | **Yes, partially.** Decontaminated by fitting the PCA on a random half of each rater's trials and testing DIF on the held-out half. Reported separately and never as the headline. |

## Effect sizes

Δ*R*²(Nagelkerke) between the matched-only model and the model with group and
group×ability. Two published classifications, both reported because they disagree:

- **Zumbo & Thomas (1997):** negligible < 0.13, moderate 0.13–0.26, large ≥ 0.26.
- **Jodoin & Gierl (2001):** negligible < 0.035, moderate 0.035–0.070, large ≥ 0.070.

Zumbo–Thomas is widely regarded as too lenient; Jodoin–Gierl is the one to read.
Reporting both is the honest move, and the classification you prefer is stated
rather than chosen after seeing the answer.

Significance uses **Wald tests with rater-clustered standard errors**. Trials are
nested within rater, so an unclustered likelihood-ratio test would treat ~220,000
correlated trials as independent and declare everything significant. At this n
almost any group difference clears p < .001, which is exactly why the effect size,
not the p-value, is the reported quantity.

## The transfer demonstration

DIF says the items function differently. It does not say anyone should care. The
transfer test puts a number on the cost:

1. Split raters into groups A and B on a grouping variable.
2. The labelling function `f_A(clip)` = majority label of that clip among group-A
   raters. This is exactly what a model trained to convergence on group-A labels
   would output, so no audio model is needed to measure the ceiling of transfer.
3. Score `f_A` against the group-B consensus.
4. **Null:** randomly reassign group labels, preserving group sizes, and recompute.
   The permutation null is matched on panel size by construction, so any degradation
   above it is attributable to the grouping and not to smaller panels — which is the
   confound that makes naive A/B splits uninterpretable.

Report observed transfer accuracy, the permutation null distribution, the gap, and
a permutation p-value, with clip-level bootstrap CIs and n.
