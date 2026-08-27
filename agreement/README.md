# agreement/

How much do CREMA-D's 2,443 crowd raters agree with each other? Not much, and
in the audio-only condition this project cares about, barely at all.

    .venv/bin/python agreement/agreement.py     # self-checks the estimators
    .venv/bin/python agreement/run.py           # writes out/agreement.json + out/per_rater.csv

`agreement.py` holds the estimators; `run.py` is the analysis. Every number
below is in `out/agreement.json` with its n. CIs are 95% percentile bootstrap,
2,000 replicates, **resampling units (clips), not individual ratings** — ratings
within a clip are not independent, which is the whole thing being measured.

Estimators are validated, not asserted: `agreement.py`'s `__main__` checks
Krippendorff's alpha against the `krippendorff` package on ragged and balanced
designs and Fleiss' kappa against `statsmodels`, and checks that the
coincidence matrix reproduces observed disagreement.

## The headline

Krippendorff's alpha, nominal, 6 categories, one unit per clip per condition.

| condition | alpha | 95% CI | clips | ratings |
|---|---|---|---|---|
| **audio only** | **0.265** | 0.259 – 0.272 | 7,442 | 73,253 |
| visual only | 0.447 | 0.440 – 0.454 | 7,442 | 73,191 |
| audiovisual | 0.486 | 0.480 – 0.493 | 7,442 | 73,242 |
| all three pooled | 0.412 | 0.407 – 0.416 | 22,326 | 219,686 |

Krippendorff's own thresholds are alpha >= 0.800 for firm conclusions and
>= 0.667 for tentative ones. Audio-only is at **one third** of the tentative bar.

Paired bootstrap over the 7,442 clips, every one of which was rated in all three
conditions, so the contrast is within-clip:

| contrast | delta alpha | 95% CI |
|---|---|---|
| audio − audiovisual | −0.221 | −0.228 – −0.213 |
| audio − visual | −0.182 | −0.191 – −0.173 |
| visual − audiovisual | −0.039 | −0.046 – −0.032 |

Stripping the face costs about five times what stripping the voice costs. Every
speech-emotion-recognition benchmark on CREMA-D runs on the audio-only column,
which is the column where the labels are worst.

## Alpha vs Fleiss' kappa

Fleiss' kappa needs every unit rated by the same number of raters. CREMA-D cells
hold 6–12, so kappa is undefined on the corpus as it stands. Computed on the
exactly-10 subset, with alpha recomputed on that same subset:

| condition | Fleiss kappa (n=10 cells) | alpha, same cells | alpha, all cells | cells kept |
|---|---|---|---|---|
| audio | 0.2649 | 0.2649 | 0.2655 | 5,044 / 7,442 |
| visual | 0.4475 | 0.4476 | 0.4473 | 5,011 / 7,442 |
| audiovisual | 0.4883 | 0.4884 | 0.4865 | 5,041 / 7,442 |

**They agree to four decimals.** That is the honest result, so it is what is
reported. The two statistics *can* differ: alpha's expected disagreement carries
an (n−1) finite-sample correction that kappa's does not, and alpha weights each
unit by its rater count. Here the design is close enough to balanced that the
correction is negligible. The difference that actually matters is availability,
not value — kappa forces you to discard 32% of the cells, and the discarded ones
are systematically the thinly-rated ones. Alpha is the right tool because it is
*defined* here, not because it gives a kinder number.

## Per intended-emotion class

`alpha_local` uses the subset's own response marginals. It is the number people
usually report and it is **not comparable across classes**: restricting units to
one intended emotion also restricts the marginals, which shrinks expected
disagreement and drags alpha down for reasons unrelated to difficulty.
`alpha_global` holds expected disagreement at the full-condition value.
Neutral is the clearest case — local 0.067, global 0.474, from the same ratings.

Audio only, 1,271 clips per class (1,087 for neutral):

| intended | alpha_local | 95% CI | alpha_global | share of raters saying it |
|---|---|---|---|---|
| anger | 0.197 | 0.185 – 0.211 | 0.296 | 51.6% |
| fear | 0.208 | 0.196 – 0.222 | 0.227 | 31.4% |
| happy | 0.194 | 0.181 – 0.208 | 0.240 | 28.3% |
| disgust | 0.146 | 0.136 – 0.156 | 0.142 | 28.3% |
| sad | 0.120 | 0.108 – 0.131 | 0.242 | 24.9% |
| neutral | 0.067 | 0.056 – 0.080 | 0.474 | 75.4% |

Audio-only, **not one of the six emotions clears alpha_local = 0.21**. Anger is
the only one raters recover more than half the time. Sad is recovered a quarter
of the time — barely above the 16.7% you get by guessing.

Full three-condition tables are in `out/agreement.json` under `alpha_by_class`.

## What raters actually confuse

The coincidence matrix — how often two responses land on the *same clip* — never
mentions the actor's intent, so it answers "what do raters confuse" without
smuggling in the label. Top pairs by share of off-diagonal mass:

| audio only | visual only | audiovisual |
|---|---|---|
| neutral / sad — 16.7% | neutral / sad — 13.3% | neutral / sad — 13.6% |
| disgust / neutral — 15.6% | anger / disgust — 12.0% | fear / neutral — 13.4% |
| fear / neutral — 14.6% | disgust / neutral — 11.2% | anger / disgust — 13.2% |
| anger / disgust — 10.7% | fear / neutral — 9.7% | disgust / neutral — 10.5% |
| anger / neutral — 9.4% | fear / sad — 9.5% | fear / sad — 10.4% |

Four of the audio-only top five involve **neutral**. The response-given-intent
matrix says why: in the voice-only condition, emotion collapses into "nothing".

Rows are the emotion the actor was directed to portray, columns are what raters
said, audio only:

| intended \ said | anger | disgust | fear | happy | neutral | sad |
|---|---|---|---|---|---|---|
| anger | **0.516** | 0.216 | 0.053 | 0.020 | 0.183 | 0.011 |
| disgust | 0.120 | **0.283** | 0.094 | 0.028 | *0.377* | 0.098 |
| fear | 0.064 | 0.066 | **0.314** | 0.030 | *0.386* | 0.139 |
| happy | 0.071 | 0.078 | 0.084 | **0.283** | *0.449* | 0.033 |
| neutral | 0.041 | 0.061 | 0.052 | 0.021 | **0.754** | 0.072 |
| sad | 0.020 | 0.071 | 0.117 | 0.013 | *0.529* | **0.250** |

For **sad**, **happy**, **fear** and **disgust**, the single most common
audio-only response is *neutral*, not the intended emotion. A sad voice is
called neutral more than twice as often as it is called sad. Anger is the one
emotion that reliably survives the loss of the face.

This is not a subtle statistical artefact — it is the plurality response for
four of six classes, over ~12,500 ratings each.

## Per-rater reliability

Each rating is scored against the **leave-one-out** consensus of the other raters
on that clip-condition cell; ties split proportionally. Leave-one-out matters:
with 6–12 raters a cell, including a rater in the consensus they are scored
against hands them a free vote, and the bias is worst in exactly the thinnest
cells.

| | value |
|---|---|
| raters | 2,443 |
| mean agreement with LOO consensus | 0.627 (sd 0.087) |
| 5th / 25th / 50th / 75th / 95th pct | 0.478 / 0.576 / 0.639 / 0.687 / 0.750 |
| min / max | 0.095 / 0.839 |
| mean expected under marginal-matched random responding | 0.207 |
| by condition | audio 0.559, visual 0.645, audiovisual 0.676 |

The null is deliberately not 1/6. A rater who clicked with no signal at all but
with realistic button-press base rates would still match the consensus about 21%
of the time. Each rater gets a one-sided exact binomial test against their own
such expectation (it depends on which clips they saw), Benjamini-Hochberg
corrected across all 2,443.

**Only 10 raters (0.41%) fail to beat chance.** Five sit below their chance
expectation on the point estimate. This is a clean crowd.

### The negative finding that matters

Dropping all 10 at-chance raters barely moves anything:

| condition | alpha before | alpha after | ratings dropped |
|---|---|---|---|
| audio | 0.2655 | 0.2677 | 293 of 73,253 |
| visual | 0.4473 | 0.4507 | 270 of 73,191 |
| audiovisual | 0.4865 | 0.4893 | 284 of 73,242 |

+0.002 on audio-only. **You cannot clean your way out of this.** The
disagreement is not a spam problem or a quality-control problem — it is 2,433
raters who are individually performing well above chance and still not
converging on the same label. The at-chance raters are also not obviously lazy:
their median response time is 3.9 s against 4.3 s for everyone else.

## Does exaggeration help? (testing the framing before asserting it)

The project's framing is that CREMA-D is *acted, exaggerated* emotion and
therefore the optimistic case, so natural speech must be worse. That is a
testable claim, not a rhetorical one, and CREMA-D contains the experiment.

Sentence IEO ("It's eleven o'clock") — and only IEO — was recorded at three
directed intensity levels, for the five non-neutral emotions, by all 91 actors:
91 x 5 x 3 = **455 clips per level**, balanced on actor and emotion. Every other
clip is level "unspecified". The contrast is therefore within-sentence and
within-actor by construction.

Audio only, IEO clips, ~4,480 ratings per level:

| directed intensity | alpha | 95% CI | share matching intent |
|---|---|---|---|
| low | 0.168 | 0.147 – 0.189 | 21.9% |
| medium | 0.266 | 0.243 – 0.287 | 31.5% |
| high | 0.348 | 0.323 – 0.372 | 45.6% |

Monotone, and the low/high CIs do not overlap. Paired bootstrap over the 455
(actor x emotion) cells, so the same actor performing the same emotion is
compared with themselves: **high − low = +0.179 (95% CI 0.151 – 0.210)**. Same
direction in the other two conditions: visual +0.125 (0.088 – 0.161),
audiovisual +0.200 (0.164 – 0.235).

**The framing holds.** Turning the performance up demonstrably buys agreement,
which means the corpus average is propped up by deliberate exaggeration and
un-exaggerated speech sits at the low end — audio-only alpha 0.168.

Two things this does *not* license saying:

- It does not measure natural speech. CREMA-D contains no spontaneous emotion.
  "Natural speech is worse" is an extrapolation from a monotone trend inside an
  acted corpus, and should be labelled as one. **Unverified against any
  spontaneous-speech corpus.**
- "Low" intensity is still a deliberate portrayal by an actor told which emotion
  to convey. It is quiet acting, not absence of acting.

What is safe to say: even at **maximum deliberate exaggeration**, audio-only
alpha is 0.348 — still less than half of Krippendorff's 0.800 bar and below his
0.667 tentative bar. The best case here is not a good case.

## Caveats

- Alpha treats the six emotions as unordered and equally distant. Neutral/sad and
  neutral/disgust are almost certainly not equally distant perceptually; a weighted
  metric would give a different (probably higher) number. Nominal is the right
  default because the benchmarks it is being compared against score exact-match
  accuracy over the same six unordered classes.
- `intended_emotion` is the actor's *direction*, not a ground truth about the
  clip. Where this document says "hit rate" it means agreement with the
  director's intent, which is a different quantity from inter-rater agreement and
  is never used as the reliability estimate.
- 50 of 2,443 raters have partial sessions (see `data/README.md`). They are kept.
  Alpha is defined under missing data; that is one of the reasons for using it.
- `actor_race` / `actor_ethnicity` are the dataset's coding of the actors and are
  not analysed here. Rater demographics were not collected at all, so nothing in
  this directory can speak to whose idiosyncrasy the labels encode.
