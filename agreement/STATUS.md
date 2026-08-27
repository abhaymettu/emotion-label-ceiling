# STATUS — ingest + agreement (agent A), 2026-08-26

Both deliverables are finished, verified, and pushed. Every number below came
from a run that completed; nothing here is projected or estimated.

## Data

`data/ratings_long.parquet` — **219,686 rows**, one per (rater, clip, modality).
`data/clips.parquet` — **7,442 rows**, clip grain with per-modality consensus.
Schema follows `CONTRACT.md`. Only `response_intensity` is ever null (2 rows).

**7,442 clips · 2,443 raters · 91 actors · 6 emotions · 3 presentation conditions.**
Every clip rated in all three; 6–12 raters per clip-condition cell.

Verified against the dataset's own R pipeline: `validate_against_authors.py`
reproduces **all 22,326 vote-count cells and all three majority-vote columns
exactly**, on 212,000 votes.

## Headline: Krippendorff's alpha (nominal, 6 classes)

95% percentile bootstrap, 2,000 reps, resampling clips.

| condition | alpha | 95% CI | clips | ratings |
|---|---|---|---|---|
| **audio only** | **0.265** | 0.259 – 0.272 | 7,442 | 73,253 |
| visual only | 0.447 | 0.440 – 0.454 | 7,442 | 73,191 |
| audiovisual | 0.486 | 0.480 – 0.493 | 7,442 | 73,242 |
| pooled | 0.412 | 0.407 – 0.416 | 22,326 | 219,686 |

Krippendorff's thresholds: >= 0.800 firm, >= 0.667 tentative. Audio-only is at a
third of the tentative bar.

Paired within-clip: audio − audiovisual = **−0.221** (CI −0.228 to −0.213).
Fleiss' kappa on the balanced n=10 subset agrees with alpha to four decimals
(audio 0.2649 vs 0.2649); it is simply undefined on 32% of the cells.

## The four findings

1. **Voice-only emotion collapses into "neutral".** For sad, happy, fear and
   disgust, the plurality audio-only response is *neutral*, not the intended
   emotion. Sad clips are called neutral 52.9% and sad 25.0%. Anger (51.6%) is
   the only emotion that survives losing the face.

2. **The majority-vote label is thin.** Audio-only: 4.0% of clips unanimous,
   21.1% have no majority at all, 8.5% are ties broken arbitrarily, and the
   consensus disagrees with the actor's intent on 54.8% of clips.

3. **You cannot clean your way out of it.** Only 10 of 2,443 raters (0.41%) fail
   a BH-corrected binomial test against chance. Dropping all 10 moves audio-only
   alpha from 0.2655 to 0.2677. Mean per-rater agreement with the leave-one-out
   consensus is 0.627 (sd 0.087) against a 0.207 chance expectation. The raters
   are good; the construct is not.

4. **The acted-is-the-optimistic-case framing holds — tested, not asserted.**
   Sentence IEO was recorded at three directed intensities by all 91 actors, so
   the contrast is within-actor. Audio-only alpha: **0.168 low → 0.266 medium →
   0.348 high**; paired high − low = **+0.179** (CI 0.151 – 0.210). Exaggeration
   buys agreement, so un-exaggerated speech sits at the low end. Even at maximum
   deliberate exaggeration, alpha is 0.348. *Extrapolation to spontaneous speech
   is unverified — CREMA-D contains none.*

## Side finding

The published CREMA-D vote tables exclude 3.5% of responses (every response whose
first emotion click took >10 s) and the dataset README never says so. Audio-only
loses 6.40% against ~2% for the other conditions. It lifts audio-only alpha from
0.2655 to **0.2811** — small, systematic, flattering. Carried as
`authors_excluded`, not applied.

## Reproduce

    ./data_ingest/fetch.sh && .venv/bin/python data_ingest/build_ratings.py
    .venv/bin/python data_ingest/validate_against_authors.py
    .venv/bin/python agreement/agreement.py      # validates the estimators
    .venv/bin/python agreement/run.py            # -> agreement/out/agreement.json

## Known gaps

- 184 responses short of 2,443 x 90; 50 raters have partial sessions. In the
  source CSV, documented, not imputed.
- `./data_ingest/fetch.sh --audio` was still running at hand-off (~2,800 of 7,442
  WAVs in `data/audio/repo/AudioWAV/`, gitignored). Re-run it; it resumes.
- Alpha treats the six emotions as equidistant. A weighted metric would likely
  give a higher number; nominal matches how the benchmarks score.
