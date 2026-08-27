# data/

Everything here is derived from CREMA-D by `data_ingest/`. Nothing is hand-edited.

    ./data_ingest/fetch.sh            # CSVs -> data/raw/  (gitignored)
    ./data_ingest/fetch.sh --audio    # + ~600MB of WAVs -> data/audio/  (gitignored)
    .venv/bin/python data_ingest/build_ratings.py

**Source:** [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), master branch, fetched 2026-08-26.
**Paper:** Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R. *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Trans. Affective Computing 2014;5(4):377-390.
**Licence:** ODbL 1.0 (database) / DbCL 1.0 (contents).

The file that matters is `finishedResponses.csv`: **one row per individual rater
response**, not a majority-vote label. That is the whole reason this project uses
CREMA-D. Almost every published CREMA-D benchmark trains against
`processedResults/tabulatedVotes.csv` (the aggregated vote) and never looks at
the 219,686 raw judgements underneath it.

## `ratings_long.parquet` — 219,686 rows x 18 columns

One row per (rater, clip, presentation modality). 7,442 clips x 2,443 raters x 3 modalities,
every clip rated in all three conditions.

| Column | Type | Source | Notes |
|---|---|---|---|
| `clip_id` | string | `clipName` | e.g. `1001_DFA_ANG_XX`. `{actor}_{sentence}_{emotion}_{intensity}`. |
| `rater_id` | string | `localid` | Crowd worker. Stable across that worker's ~90 responses. |
| `presented_modality` | category | `queryType` | `audio` (voice only, qT=1), `visual` (face only, qT=2), `audiovisual` (qT=3). |
| `response_emotion` | category | `respEmo` | Forced choice: `anger`/`disgust`/`fear`/`happy`/`neutral`/`sad`. Recoded from CREMA-D's `A/D/F/H/N/S`. |
| `response_intensity` | float | `respLevel` | Rater's intensity slider, 1-100. **Null on 2 rows** (see drops). Only column permitted to be null. |
| `intended_emotion` | category | filename field 3 | What the actor was directed to portray. Same six-value vocabulary. |
| `intended_intensity` | category | filename field 4 | `low` / `medium` / `high` / `unspecified`. 81.66% of clips are `unspecified`; the other three split 6.11% each. |
| `actor_id` | string | filename field 1 | 4-digit, 91 actors. |
| `actor_sex` | category | `VideoDemographics.csv` | `male` / `female`, lowercased. 48 / 43. |
| `actor_age` | int | `VideoDemographics.csv` | 20-74 at recording. |
| `actor_race` | category | `VideoDemographics.csv` | African American / Asian / Caucasian / Unknown. **The dataset's coding of the actor, not a fact about the person.** |
| `actor_ethnicity` | category | `VideoDemographics.csv` | Hispanic / Not Hispanic. Same caveat. |
| `sentence_id` | category | filename field 2 | One of 12 fixed sentences (`IEO`, `TIE`, `IOM`, ...). |
| `response_time_ms` | int | `ttr` | Time to respond. Useful for spotting click-through raters. |
| `num_tries` | int | `numTries` | Extra emotion clicks before settling. 0 on 99.1% of rows. |
| `session_num` | int | `sessionNums` | Rating session. Effectively 1:1 with `rater_id`. |
| `question_num` | int | `questNum` | Position within that modality block, 1-30. |
| `log_pos` | int | `pos` | Raw log order for that participant. Kept only to make duplicate resolution auditable. |
| `authors_excluded` | bool | derived | True on the 7,687 rows (3.50%) that CREMA-D's own R pipeline throws away before publishing its vote tables. See below. |

Columns past `sentence_id` are extras beyond `CONTRACT.md`; the contract columns are all present and named as specified.

## `clips.parquet` — 7,442 rows x 23 columns

Clip grain. Carries the clip's identity and demographics, `n_ratings` plus
`n_ratings_{audio,visual,audiovisual}`, and for each modality
`consensus_{mod}` (modal response), `agreement_{mod}` (modal share of that
modality's votes) and `consensus_tied_{mod}`. Ties are broken by a fixed
emotion order so the column is deterministic — **check `consensus_tied_*`
before treating a consensus label as meaningful.** `wav_file` is the basename
under `data/audio/repo/AudioWAV/`.

## Dropped and altered rows — the complete list

Two rows dropped out of 219,688. Both are duplicate `(rater, clip, modality)` cells:

| Rule | Rows affected | Rows dropped | What we did |
|---|---|---|---|
| Duplicate `(rater, clip, modality)` | 4 (2 cells) | 2 | Kept the last response by `log_pos`. One pair is a byte-identical replay of the same log line (`SSI_1121558943` / `1034_IEO_HAP_MD`); the other is a rater who changed their answer from happy to anger (`SSI_1132064474` / `1059_IEO_HAP_MD`, `numTries=1` on the second). The source file is documented as holding the *final* response, so last-by-position is the intended semantics. |
| Malformed `respLevel` | 3 (2 surviving) | 0 | `ans` was e.g. `"H"` instead of `"H_54"`, so the intensity slider value is missing and the emotion letter landed in the level column. The emotion is intact, so the row is kept and `response_intensity` is null. |

Nothing else was dropped. No row was dropped for a missing field.

**`dispVal` is not used.** It is NA on 540 rows. It is fully redundant with
`dispLevel`, which is never missing and matches the filename on 100% of rows, so
`intended_intensity` is read off the filename and no row is lost to it.

**The CREMA-D README's column list is wrong about two columns.** It describes
`dispVal` as "the displayed value" and `dispLevel` as "a numeric representation
(20/50/80)". In the actual file it is the other way round: `dispVal` holds
20/50/80 and `dispLevel` holds `L/M/H/X`. We follow the data, and we cross-check
both against the filename.

## The authors' published vote tables exclude 3.5% of responses, undocumented

`processFinishedResponses.R` drops every response whose **first emotion click
took over 10 seconds**, matched on `sessionNums*1000 + queryType*100 + questNum`
against `finishedEmoResponses.csv`. 7,687 of 219,688 responses. Nothing in the
dataset README says so, and `processedResults/tabulatedVotes.csv` and
`summaryTable.csv` — the labels most published CREMA-D work scores against —
are computed after it.

We do not apply it. `ratings_long.parquet` flags it as `authors_excluded` so
either subset is one filter away, and

    .venv/bin/python data_ingest/validate_against_authors.py

reproduces **all 22,326 of their vote-count cells and all three majority-vote
columns exactly**, on 212,000 votes, once the filter is applied. That is an
independent check of this whole ingest against the authors' own R code, and it
is how the filter was found: without it, 6,131 cells disagree, always with fewer
votes on their side.

The dropped responses are not spread evenly — audio-only loses 6.40% against
2.1% and 2.0% for the other conditions. See `agreement/README.md` for what that
does to alpha (it raises it, by 0.016 on audio-only).

## Reconciliation against the dataset's own published counts

Every claim below is asserted in code (`PAPER` in `data_ingest/build_ratings.py`);
`data/ingest_report.json` carries the machine-readable version.

Reproduce **exactly**: 7,442 clips, 91 actors (48 male / 43 female), 2,443 raters,
12 sentences, actor ages 20-74, and 22,326 = 7,442 x 3 clip-modality cells with
no cell missing. `SentenceFilenames.csv`'s 7,442 filenames are exactly the set of
rated clips.

### Discrepancy 1 — 184 responses short (reported, not silently accepted)

The README says "2443 participants each rated 90 unique clips", which implies
2,443 x 90 = **219,870** responses. The source CSV has **219,688**; after
duplicate resolution we have **219,686**. Shortfall: **184**.

This is in the source file, not introduced by our parsing. 2,393 raters have
exactly 90 responses; **50 raters have fewer** (minimum 37), and 65
rater-by-modality cells hold something other than 30 responses. The "each rated
90" claim is an approximation — 98.0% of raters did. Nothing here is dropped or
imputed; partial sessions are kept as-is, and any analysis that needs balanced
raters must filter on `rater_id` counts itself.

### Discrepancy 2 — the ">7 ratings" claim is understated

The README says "95% of the clips have more than 7 ratings". Observed:
**99.61%** of clip-modality cells have more than 7, and **100%** of clips have
more than 7 ratings in total (26-32 per clip). Their claim holds, with room to
spare. Ratings per clip-modality cell:

| ratings | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|
| cells | 12 | 76 | 436 | 4,584 | 15,096 | 2,086 | 36 |

## First thing you notice

Raters agree with the actor's *intended* emotion on:

| modality | rater responses matching intended emotion | n |
|---|---|---|
| audio only | **39.2%** | 73,253 |
| visual only | 56.9% | 73,191 |
| audiovisual | 62.5% | 73,242 |

Chance is 16.7%. That table is agreement with the *director's intent*, which is
not the same thing as inter-rater agreement. For the reliability statistics that
actually matter, see [`agreement/README.md`](../agreement/README.md) — the short
version is Krippendorff alpha 0.265 audio-only, 0.447 visual, 0.486 audiovisual.

## Audio

`./data_ingest/fetch.sh --audio` sparse-clones AudioWAV from the GitLab mirror
into `data/audio/repo/AudioWAV/` (gitignored). 7,442 files, 16 kHz mono PCM,
~600MB, and about the same again in `.git/lfs`. `clips.parquet.wav_file` is the
basename. `raw.githubusercontent.com` will not do: it serves 130-byte git-lfs
pointer stubs for these, so anything that fetches WAVs from there is silently
downloading text files.

Four clips are known-broken upstream and are *still in* both parquets, because
the raters rated whatever was actually played to them and dropping the ratings
would misrepresent the rating data. Excluding them is an audio-modelling
decision, not an ingest one: `1076_MTI_NEU_XX` and `1076_MTI_SAD_XX` (near-empty),
`1064_TIE_SAD_XX` (no duration), `1064_IEO_DIS_MD` (a 1-minute file containing
every emotion for that sentence).

### The mirror is byte-identical to the official repo on the audio — verified, not assumed

The WAVs come from an **unofficial** mirror, `gitlab.com/cs-cooper-lab/crema-d-mirror`,
not from `github.com/CheyneyComputerScience/CREMA-D`. For a repo whose whole argument is
provenance, that has to be checked rather than trusted, so it was, on **2026-08-26**:

    GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout --depth 1 \
      https://github.com/CheyneyComputerScience/CREMA-D.git official
    # every AudioWAV blob in each repo is a git-lfs pointer carrying the sha256 of the
    # real file, so the two file lists can be compared without downloading either one
    git -C <repo> ls-tree -r HEAD --format='%(objectname) %(path)' | grep ' AudioWAV/' ...
    # then, separately, hash what actually landed on disk
    shasum -a 256 data/audio/repo/AudioWAV/*.wav

Three checks, all of them exhaustive — every file, not a sample:

| check | result |
|---|---|
| AudioWAV file count, mirror vs official | 7,442 = 7,442 |
| (filename, git-lfs sha256) pairs, mirror vs official | **all 7,442 identical** |
| sha256 of the 7,442 WAVs actually on disk vs the official pointers | **all 7,442 identical** |

So the audio this repo models is the official audio, byte for byte. The rating CSVs are
identical too: `finishedResponses.csv`, `finishedEmoResponses.csv`,
`finishedResponsesWithRepeatWithPractice.csv`, `SentenceFilenames.csv`,
`VideoDemographics.csv` and all four R scripts have the same git blob hashes in both
repos. (They are fetched from `raw.githubusercontent.com` anyway, so the ingest never
depends on the mirror for them.)

The mirror does differ from the official repo, in three ways, none of which touches
anything used here:

- **It is missing `processedResults/`** — no `summaryTable.csv`, no `tabulatedVotes.csv`.
  Those are the published vote tables most CREMA-D papers score against, and they are the
  files the 3.5% filter above is applied to. This ingest pulls them from
  `raw.githubusercontent.com` (official), so `validate_against_authors.py` is still
  checking against the authors' real output.
- **`docs/` is renamed `public/`** and `README.md` and `public/README.md` are edited —
  GitLab Pages housekeeping. The mirror's one visible commit is `Delete index.html`.
- **One extra file:** `VideoFlash/1015_DFA_ANG_XX.mp4`, which the official repo does not
  have. Video is not used by this project.

Re-run the check any time; it needs no LFS download and takes about a minute.
