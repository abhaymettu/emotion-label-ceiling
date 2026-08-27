# emotion-label-ceiling

**On audio-only CREMA-D, the maximum accuracy any model can reach against the crowd
consensus label is 72.7% (95% CI 72.3–73.2, n = 7,442 clips, 2,443 raters, median 10
raters per clip). Four published speaker-independent results sit above it: 76.75,
74.50, 73.83 and 73.34.**

That is not automatically misconduct, and the rest of this repo is about why. The
short version: almost nobody says which label they scored against, and if they scored
against the actor's *intended* emotion rather than the crowd's, no such ceiling applies
to them — but then their number cannot be compared to the 40.9% humans get.

---

## What CREMA-D is, and why it is the optimistic case

7,442 clips. 91 actors, each reading 12 fixed sentences, each **directed to portray**
one of six emotions: anger, disgust, fear, happy, neutral, sad. 2,443 crowd raters,
~90 clips each, forced choice among the same six. Every rater saw all three
presentation conditions — audio-only, visual-only, audiovisual — about 30 clips each.
219,686 ratings total.

Two things follow.

**This is acted emotion, so it is the easy case.** The actor knew what they were
performing. There is a recorded intended label. Spontaneous affect has no such
ground truth and is harder in every respect. Whatever ceiling shows up here is
generous relative to real speech.

**There are two different labels in this dataset and papers rarely say which they
used.** The filename encodes the intended emotion; `summaryTable.csv` ships separate
crowd majority votes. The distinction decides whether a reliability ceiling applies
at all.

## What the raters actually agree on

Computed here, per modality, n = 7,442 clips:

| | audio | visual | audiovisual |
|---|---|---|---|
| Krippendorff's α (nominal) | **0.266** | 0.447 | 0.487 |
| Pairwise agreement between two raters | 0.455 | 0.546 | 0.579 |
| Crowd majority reproduces the actor's intent | **0.440** | 0.683 | 0.743 |
| Clips where the majority is a tie | 8.6% | 6.3% | 5.7% |

The bottom row is a replication check. Cao et al. (2014) report 41% / 64% / 72% for
the same quantity in their Table 7. We get 44.0 / 68.3 / 74.3 using a plain
tie-broken argmax where they used a binomial-majority rule that sets aside 8–13% of
clips as ambiguous. Same numbers, and the small gap is the tie handling.

**From voice alone, a crowd of ten people recovers what the actor was told to
perform 44% of the time.** Chance is 17%.

## The ceiling

Full derivation and every assumption: [`ceiling/DERIVATION.md`](ceiling/DERIVATION.md).

The benchmark label is the majority vote of a panel of *R* raters. That panel is a
sample, so the label $Y$ is a random variable, and a deterministic model $f(X)$ is
bounded by the Bayes rate against it:

$$C(R) = \mathbb{E}_X\Big[\max_y \Pr(Y = y \mid X)\Big]$$

We estimate $\Pr(Y \mid X = x_i)$ as the posterior predictive of the consensus label
under a Dirichlet-multinomial model of the per-clip response counts, shrinking toward
the marginal profile of the clip's intended class, with the concentration fitted by
maximum likelihood. The max is taken **after** integrating over the posterior, and
the Monte Carlo is cross-fitted (half the draws pick the argmax, half estimate its
probability). Both details matter: the naive estimator that takes the max of six
noisy empirical proportions returns **0.816** on audio where the corrected one
returns 0.727. That 9-point gap is pure winner's curse.

| modality | R | **ceiling** | 95% CI | split-half lower bound | naive (biased, unused) |
|---|---|---|---|---|---|
| audio | 10 | **0.727** | [0.723, 0.732] | 0.644 | 0.816 |
| visual | 10 | **0.797** | [0.792, 0.800] | 0.729 | 0.860 |
| audiovisual | 10 | **0.828** | [0.825, 0.833] | 0.764 | 0.878 |
| all pooled | 30 | 0.866 | [0.863, 0.870] | 0.813 | 0.905 |

The split-half column is an assumption-light cross-check: an oracle that sees half a
clip's raters, scored against the majority of the other half. It shares none of the
Dirichlet machinery and is conservative by construction (both sides see R/2). The
model-based estimate sits between it and the naive one, which is where it should be.

**The ceiling is a property of the protocol, not of emotion.** It rises with panel
size — on audio: 0.61 at R=3, 0.73 at R=10, 0.79 at R=31, 0.82 at R=201 — and tends
to 1 as R grows. A model that reaches $C(R)$ has learned the modal percept of a
ten-person crowd. That is a real thing. It is not "the emotion."

## Against published SOTA

Every number in [`ceiling/sota.csv`](ceiling/sota.csv) was read out of the paper's own
abstract or results table. Provenance, rejected numbers and the two values a search
engine got wrong: [`ceiling/SOURCES.md`](ceiling/SOURCES.md).

**Audio-only, 6-class, speaker-independent splits. Ceiling = 0.727.**

| system | reported | vs ceiling | source |
|---|---|---|---|
| EmoBox Whisper large v3 | 76.75 UA | **+4.1** | [arXiv:2406.07162](https://arxiv.org/abs/2406.07162) |
| EmoBox WavLM large | 74.50 UA | **+1.8** | same |
| EmoBox HuBERT large | 73.83 UA | **+1.1** | same |
| WavLM-Plus (in HiCMAE) | 73.34 UAR | **+0.6** | [arXiv:2401.05698](https://arxiv.org/abs/2401.05698) |
| EmoBox HuBERT base | 71.13 UA | −1.6 | [arXiv:2406.07162](https://arxiv.org/abs/2406.07162) |
| HiCMAE-B (audio) | 71.11 UAR | −1.6 | [arXiv:2401.05698](https://arxiv.org/abs/2401.05698) |
| EmoBox WavLM base | 69.64 UA | −3.1 | [arXiv:2406.07162](https://arxiv.org/abs/2406.07162) |
| Cao et al. 2014, human crowd majority | 41.0 | −31.7 | [PMC4313618](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/) |

**Audiovisual. Ceiling = 0.828.** The single most informative row in this repo:
**VAVL** ([Goncalves et al. 2025](https://arxiv.org/abs/2305.07216)) is the *only*
paper of the fourteen we checked that states it scores against the crowd consensus
("the perceived emotions from the audio-visual modality"). It reports **0.826 ± .015
F1-micro**. The ceiling is **0.828 [0.825, 0.833]**.

The one result we can verify is measured against the same target as our ceiling lands
exactly on it, and does not exceed it.

Everything reported above the audiovisual ceiling — 89.49, 85.06, 84.89, 84.57,
83.70 — is label-target **unspecified**. So is every audio number in the table above.
Twelve of fourteen papers never say.

**Visual-only. Ceiling = 0.797.** Published: 77.33, 77.31, 77.25. All below. No
violation.

### What this does and does not show

The honest reading, in order:

1. **Where the label target is verified, nothing exceeds the ceiling.** One paper,
   sitting on it. That is a negative result for the strong version of the thesis and
   it stays stated as one.
2. **Four audio numbers exceed the ceiling, but their label target is unknown.** If
   they scored against the intended label, no reliability ceiling applies and they
   are doing nothing wrong — the intended label is deterministic given the clip. If
   they scored against consensus, the ceiling says the excess is not emotion.
   **We cannot tell which, and neither can a reader of those papers.** That is the
   actual finding: the benchmark does not record what it is measuring.
3. **The interesting gap is not model-vs-ceiling, it is model-vs-human.** Machines
   are reported at 76.5% audio against a target the entire human crowd reproduces
   41% of the time. Whatever those models are recovering from the waveform, it is
   not something ten listeners can hear.

We are also explicit that the *broad* claim here has been refuted three times, and
we do not make it. "Inter-annotator agreement caps model accuracy" is wrong —
see [Reidsma & Carletta 2008](https://aclanthology.org/J08-3001/),
[Boguslav & Cohen 2017](https://pubmed.ncbi.nlm.nih.gov/29295103/), and
[Richie et al. 2022](https://aclanthology.org/2022.bionlp-1.26/), which is a
dedicated simulation study of exactly that proposition. Our claim is the narrow one
that survives it: a bound on **agreement with a majority vote of R noisy raters**,
which is a real bound because the evaluation target is itself a random variable.
Never a bound on accuracy against latent truth.

## Measurement invariance across annotators

Method and model justification: [`invariance/METHOD.md`](invariance/METHOD.md).

Benchmarks pool every rater's `anger` into one label as though the six options mean
the same thing to everyone. In psychometrics that is a hypothesis, not a convention.
We test it with logistic-regression DIF (Swaminathan & Rogers 1990) — six emotion
categories as items, 2,443 raters as persons, matched on **rest score** over the other
five items, keyed on the actor's intent (external to every rater, so not circular),
with rater-clustered standard errors. Mantel-Haenszel with ETS A/B/C as a
non-parametric cross-check, plus a Bock-style nominal view of *which* wrong answer
each group gives.

CREMA-D publishes no rater demographics, so groups are behavioural.

**Audio-only, n = 73,253 trials, 2,443 raters. ΔR²(Nagelkerke), Jodoin & Gierl bands:**

| grouping | worst item | ΔR² | J&G band | ETS | raw acc. gap |
|---|---|---|---|---|---|
| response speed (fast / deliberate) | anger | 0.004 | negligible | A | +4.9 pts |
| response-style extremity | neutral | 0.005 | negligible | A | −4.5 pts |
| session position (early / late) | anger | 0.006 | negligible | A | +6.3 pts |
| latent construal class, ability-residualised | **neutral** | **0.032** | negligible | **C (large)** | **+13.5 pts** |

**On every manifest behavioural grouping, DIF is negligible.** Speed, response-style
extremity and fatigue do not make the emotion items function differently. That is a
negative result and it is not softened.

The one grouping that does bite is latent: the first principal component of a rater's
6×6 confusion profile, fitted on a random half of that rater's trials and tested on
the held-out half so the grouping is not read off the trials it is tested on. Because
a confusion profile is partly just accuracy, we residualise it on the rater's own
accuracy first — **and the effect gets stronger, not weaker**, so it is construal and
not ability. ΔR² still lands in Jodoin-Gierl's "negligible" band while
Mantel-Haenszel calls it ETS C and the raw gap is 13.5 points; we report the
disagreement rather than picking the flattering metric.

### Does it cost anything — the A → B transfer test

DIF says items function differently. It does not say anyone should care. So: let
`f_A(clip)` be the majority label among group-A raters — exactly what a model trained
to convergence on group-A labels would output — and score it against the group-B
consensus. The confound is panel size, so the null is **random regroupings of the same
sizes**, matched by construction.

**Audio-only, ≥3 raters per side, 200 permutations, 300-clip bootstrap:**

| grouping | transfer acc. | 95% CI | permutation null | degradation | p | n clips |
|---|---|---|---|---|---|---|
| response speed | 0.637 | [0.628, 0.651] | 0.635 | −0.002 | 0.68 | 6,376 |
| response-style extremity | 0.633 | [0.625, 0.647] | 0.635 | +0.002 | 0.30 | 6,341 |
| session position | 0.636 | [0.625, 0.647] | 0.635 | −0.001 | 0.60 | 6,303 |
| latent construal class | 0.572 | [0.562, 0.585] | 0.635 | **+0.063** | <0.005 | 6,297 |
| ↳ ability-residualised | 0.588 | [0.579, 0.606] | 0.636 | **+0.048** | <0.005 | 5,893 |

**A labelling function fit to one rater faction loses 4.8 accuracy points on another,
beyond what a size-matched random split costs.** Against a ceiling of 72.7 that is
about a sixth of the headroom between chance and the best achievable label. Across
behavioural groupings the loss is zero.

### Which classes are least invariant

The two metrics disagree, so both are reported. By ΔR² across groupings (audio):
**neutral** (0.011), **sad** (0.008), happy (0.005), anger (0.004), fear (0.003),
disgust (0.002). By total-variation distance of the matched response distribution —
which sees *which* wrong answer was given, not just whether it was wrong: **sad**
(0.19), **disgust** (0.15), **fear** (0.14).

Sad is worst on both. It is also the class where the crowd majority reproduces the
actor's intent least often on audio (19.4%, n = 1,271 clips) against a per-class
ceiling of 0.737. Sadness in voice is the weakest part of this benchmark by every
measurement we made.

## Limitations

- **Acted emotion.** Portrayals, not felt affect, with a director's intent recorded.
  Everything here is the optimistic case. None of it transfers to spontaneous speech
  without re-measurement.
- **The label-target problem cuts both ways.** Twelve of fourteen papers do not state
  whether they scored against intended or consensus labels. We cannot assign the
  above-ceiling audio results to either explanation, and we do not.
- **Our ceiling assumes exchangeable raters within a clip — and the second half of
  this repo shows that assumption failing.** If raters are a mixture, $C(R)$ still
  bounds accuracy against a *randomly composed* panel, which is what benchmarks use,
  so the bound holds. But it stops being a property of the clip.
- **The ceiling moves with panel size and with the tie rule.** A benchmark built only
  from high-agreement clips has a much higher ceiling and a much narrower claim.
  Cao et al. report α = 0.79 on the ≥80%-agreement subset against 0.42 overall.
- **No rater demographics exist in CREMA-D**, so the invariance groupings are
  behavioural proxies. The one grouping that shows an effect is latent and
  data-derived; the held-out split and the ability residualisation address
  circularity but do not make it a substantive group like age or culture.
- **Neither of the two moves here is methodologically new.** Estimating irreducible
  error from human soft labels and auditing SOTA against it is
  [Ishida et al., ICLR 2023](https://arxiv.org/abs/2202.00395); multi-class
  estimators are benchmarked in [FeeBee](https://arxiv.org/abs/2108.13034);
  disagreement-aware re-reporting of benchmark metrics is
  [Gordon et al., CHI 2021](https://doi.org/10.1145/3411764.3445423). Applying
  measurement invariance and differential *rater* functioning to ML annotation is
  [Sachdeva et al., FAccT 2022](https://doi.org/10.1145/3531146.3533216), on a hate
  speech corpus. **We are not the first to bring invariance testing to annotation.**
  Nor is psychometric modelling of speech-emotion annotators new: Wong & Chen
  ([ASRU 2025](https://doi.org/10.1109/ASRU65441.2025.11434646)) fit a Rasch model to
  MSP-Podcast and speechocean762 to build a scalar reference that absorbs "differing
  bias between groups of annotators."

  What remains unclaimed, after screening all ~40 papers citing Sachdeva et al. and
  searching the nominal-DIF literature directly:

  1. **Nominal responses.** Every prior annotator-psychometrics result we found is
     *ordinal* — Sachdeva et al.'s many-facet Rasch, Wong & Chen's Rasch reference,
     MOS-bias work. None transfers to a six-way unordered forced choice. The machinery
     that does — Bock's nominal response model, differential distractor functioning,
     [difNLR](https://journal.r-project.org/articles/RJ-2020-014/index.html),
     multi-group latent class analysis — is mature in educational testing and, as far
     as we can find, has never been pointed at annotators.
  2. **Testing invariance rather than calibrating it away.** Wong & Chen treat
     annotator group as a *confounder to control*; we treat it as a *hypothesis to
     test*. Zero of the Sachdeva-citing papers run a DIF test on emotion or affect
     annotation in any modality.
  3. **An external criterion.** CREMA-D records what the actor was directed to
     portray. Subjective-annotation corpora have no such anchor, which is what lets us
     key DIF on something outside every rater.

  Caveat on that first point: Wong & Chen is paywalled with no preprint and we have
  read only its abstract. If its full text turns out to run group-level DIF, item 2
  weakens.
- **n is stated everywhere above and every interval is a 95% CI.** Where an interval
  is absent, the quantity is a published number we copied, not one we estimated.

## Status

Computed on real data (`data/ratings_long.parquet`, 219,686 ratings):

- [x] reliability: α, pairwise agreement, tie rates, per modality
- [x] ceiling: Dirichlet-multinomial posterior-predictive estimator, cross-fitted,
      clip-bootstrap CIs, curve in R, per modality and per emotion class
- [x] split-half assumption-light cross-check
- [x] published SOTA table, 33 rows, provenance and rejected numbers documented
- [x] LR-DIF + Mantel-Haenszel + nominal response shift, four rater groupings
- [x] A → B transfer with a size-matched permutation null

Pending:

- [ ] **figures/** — ceiling vs SOTA, DIF, transfer degradation. Not yet drawn.
- [ ] leave-one-rater-out consensus key as the DIF robustness check (currently keyed
      on intent only)
- [ ] a proper nominal response model (Bock 1972) fit rather than the matched-decile
      response-distribution comparison used here
- [ ] read the full text of Wong & Chen, ASRU 2025
      ([DOI](https://doi.org/10.1109/ASRU65441.2025.11434646)) — paywalled, no
      preprint, abstract only so far. It is the nearest prior work and the related-work
      section should not be written without it.

Every script has a runnable self-check with planted known answers:

```
.venv/bin/python ceiling/ceiling.py --check
.venv/bin/python invariance/dif.py --check
.venv/bin/python invariance/transfer.py --check
```

File ownership across the three agents working this repo: [`CONTRACT.md`](CONTRACT.md).
