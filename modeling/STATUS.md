# modeling / web — status

Last updated by agent B. Nothing here is a result yet: **no training run has finished.**

## Running right now

A watcher is polling for the audio download and will start the first fine-tune on its own:

    modeling/train_when_audio_lands.sh          # started with nohup, pid was 76533

It waits until `data/audio/repo/AudioWAV` holds all 7442 wavs (agent A's `git lfs` pull is
still filling it — 3.3k of 7.4k at the time of writing), then runs:

    .venv/bin/python modeling/finetune.py --tag wav2vec2-base-intended_emotion-actor-s0

If the watcher died, that command is safe to run by hand. It re-checks what audio exists and
prints a loud warning if any wav is missing.

Inspect:

    tail -f modeling/runs/wav2vec2-base-intended_emotion-actor-s0/train.log
    cat     modeling/runs/wav2vec2-base-intended_emotion-actor-s0/curves.csv

Everything is written incrementally, so a killed session loses at most one epoch.

| Path | Contents |
|---|---|
| `modeling/runs/<tag>/config.json` | args, seed, device, the exact actor lists per split |
| `modeling/runs/<tag>/curves.csv` | per-epoch train loss, val accuracy, seconds |
| `modeling/runs/<tag>/metrics.json` | test accuracy vs intended AND vs audio consensus, per-class precision/recall, ceiling comparison |
| `modeling/runs/<tag>/confusion.csv` | 6x6 confusion on held-out actors |
| `modeling/runs/<tag>/test_preds.csv` | per-clip predictions (feeds the web page) |
| `models/<tag>.pt` | best-val weights, gitignored |

## Finished

- `modeling/common.py` — actor-disjoint split, stratified by actor sex, 60/11/20 actors.
  `.venv/bin/python modeling/common.py` runs the assert self-check: no actor and no clip
  crosses a boundary, at 5 seeds, and it also asserts the random-clip-split baseline *does*
  leak actors, so the contrast we report is real and not an artefact.
- `modeling/finetune.py` — wav2vec2-base (or any `AutoModel` audio encoder) + mean pool +
  linear head, MPS verified working (torch 2.13, `mps.is_available()` true). Scores every run
  against both label definitions and folds in `ceiling/out/ceiling.json` automatically.
- `web/template.html` + `web/build.py` + `web/index.html` — self-contained page, 1.2 MB, no
  server and no CDN, wavs inlined as base64. Play a clip, guess, then see the actor's script,
  the crowd's actual split, and the model's answer. Rebuild after a run finishes:

      .venv/bin/python web/build.py --run wav2vec2-base-intended_emotion-actor-s0

  Until a run finishes the page honestly shows "model: not trained yet" rather than a
  placeholder number.
- `demo/` — the 12 clips the page embeds, picked to span the agreement range across all six
  emotions.

## Not done

- The rater-transfer experiment (train heads against rater group A's labels, evaluate against
  group B's). Design that was going in: freeze the **pretrained** encoder, not the fine-tuned
  one, so labels are the only thing that varies; use random rater halves as the null and a
  behavioural grouping (response-bias clustering over each rater's 30 audio judgements) as the
  contrast. Group definitions still need to match whatever agent C uses in `invariance/`.
- The random-split cautionary number (`--split random`). The code path exists and is wired up;
  no run has been done. When it is, it is a footnote, never the headline.
- Multiple seeds. `finetune.py` writes `"n_seeds": 1` and a caveat into every `metrics.json`
  for exactly this reason. One run is not a result.

## Known blocker at hand-off

The `facebook/wav2vec2-base` checkpoint had not finished downloading when this session ended:
`~/.cache/huggingface/hub/models--facebook--wav2vec2-base` held 8 KB of a ~380 MB model.
HuggingFace is reachable but was serving at **~1 KB/s** while agent A's `git lfs` pull of the
7,442 CREMA-D wavs was running. This is bandwidth contention, not a broken URL — a plain
`curl` of `config.json` returned 200 over both HTTP/1.1 and HTTP/2.

The watcher waits for the audio to finish before it starts training, so by the time the
encoder download runs the link should be free. If it is still crawling, pre-pull the encoder
by hand before anything else:

    .venv/bin/python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/wav2vec2-base')"

A forward/backward smoke test on MPS was started and had to be killed while still waiting on
that download, so **the training step has not yet been observed to run end to end**. Watch the
first epoch rather than assuming it works.

---

## Note for whoever owns `web/` (2026-08-27)

Commit `86679c2` is mine (`record the encoder's actual config and param count in every
run`) but it also carries **723 changed lines each in `web/index.html` and
`web/template.html`**. Those are not my edits. They were already sitting in the shared
index when I committed, and `git add modeling/finetune.py` does not unstage what someone
else staged. Nothing is lost, the diff is intact and pushed, but it landed under the wrong
message. Same failure mode as `5cab46a`.

Everyone on this tree: use `git commit <path> -m "..."`, not `git add <path> && git commit`.
The pathspec form commits only that path and ignores whatever else is staged.
