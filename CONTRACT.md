# CONTRACT

Three agents work this repo concurrently. This file is the interface between them.
If you need something that is not written here, add it here first, push, then build.

## File ownership

Only the owner writes to a path. Everyone else reads.

| Paths | Owner | Contents |
|---|---|---|
| `data_ingest/`, `data/*.parquet` | agent A | CREMA-D download, parse, normalization; produces `data/ratings_long.parquet` |
| `agreement/`, `modeling/`, `web/` | agent B | inter-rater agreement statistics, model training/eval, web writeup |
| `ceiling/`, `invariance/`, `figures/`, `README.md`, `CONTRACT.md` | agent C | reliability-corrected ceiling, measurement invariance / DIF, all figures, top-level writeup |

Rules:

- Never `git add -A`. Add only your own paths.
- Before every push: `git pull --rebase origin main`, then `git push origin main`.
- Do not edit another agent's file to "fix" it. Open a note in this file instead.
- `data/` is shared read. Only agent A writes parquet there.

## Primary data contract

`data/ratings_long.parquet` — one row per (rater, clip, presentation). Long format.

| Column | Type | Notes |
|---|---|---|
| `clip_id` | string | CREMA-D clip identifier, e.g. `1001_DFA_ANG_XX`. Unique per audio/video file. |
| `rater_id` | string | Stable crowd-worker identifier. Same worker across clips gets the same id. |
| `presented_modality` | category | One of `audio`, `visual`, `audiovisual`. CREMA-D ran three separate presentation conditions. |
| `response_emotion` | category | The rater's forced-choice response. One of `anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`. Lowercase. |
| `response_intensity` | category or float | Rater's intensity rating. May be null where not collected. |
| `intended_emotion` | category | The emotion the actor was directed to portray. Same 6-value vocabulary as `response_emotion`. |
| `actor_id` | string | CREMA-D actor identifier (91 actors). |
| `actor_sex` | category | `male`, `female`. |
| `actor_age` | int | Actor age in years at recording. |
| `actor_race` | category | CREMA-D's own actor demographic coding. Treat as the dataset's label, not ground truth about the person. |
| `sentence_id` | category | One of the 12 fixed sentences (e.g. `IEO`, `TIE`, `IOM`). |

Vocabulary is **lowercase, singular, exactly these six**: `anger`, `disgust`, `fear`, `happy`, `neutral`, `sad`.
Agent A normalizes CREMA-D's `ANG/DIS/FEA/HAP/NEU/SAD` codes to these strings. Downstream code assumes it.

Expected scale: ~7,442 clips, ~2,443 raters, 6 classes, ~95k+ rating rows across modalities.

### Nulls

- `response_intensity` may be null.
- Nothing else may be null. If agent A cannot resolve a field, the row is dropped and the drop is counted in the ingest log.

## Derived files agent C writes

| Path | Written by | Contents |
|---|---|---|
| `ceiling/out/ceiling.json` | ceiling | reliability estimates, ceiling estimate, bootstrap CIs, n |
| `ceiling/out/sota.csv` | ceiling | published CREMA-D accuracies, one row per paper, with citation + link |
| `invariance/out/dif.csv` | invariance | per-class DIF statistics by rater group |
| `invariance/out/transfer.json` | invariance | A→B labelling-function transfer degradation |
| `figures/*.png`, `figures/*.svg` | figures | light+dark variants |

Anything under `*/out/` is regenerable. Delete and re-run is always safe.

## Simulated data

`data/ratings_long.parquet` may not exist yet. Agent C develops against a fixture:

    .venv/bin/python ceiling/simulate.py    # writes data/SIMULATED_ratings_long.parquet

The fixture obeys this exact schema. Every script takes `--ratings PATH` and defaults to the real file,
falling back to the fixture with a loud stderr warning.

**Hard rule: no number computed from a `SIMULATED_*` file may appear in `README.md` or in `figures/`
without the word SIMULATED next to it.** Scripts stamp `"source_file"` and `"simulated": true|false`
into every JSON they emit so this is checkable, not just promised.

## Notes between agents

(append here; leave a name and a date)
