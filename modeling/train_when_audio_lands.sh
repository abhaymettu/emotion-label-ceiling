#!/usr/bin/env bash
# The audio download is still running in another agent's process. Poll until all 7442
# wavs are on disk, then start the first fine-tune. Logs to modeling/runs/<tag>/train.log.
set -u
cd "$(dirname "$0")/.."
DIR=data/audio/repo/AudioWAV
TAG=${1:-wav2vec2-base-intended_emotion-actor-s0}
for _ in $(seq 1 360); do
  n=$(ls "$DIR" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 7442 ] && break
  sleep 30
done
mkdir -p "modeling/runs/$TAG"
echo "starting with $(ls $DIR | wc -l) wavs at $(date)" > "modeling/runs/$TAG/train.log"
exec .venv/bin/python modeling/finetune.py --tag "$TAG" >> "modeling/runs/$TAG/train.log" 2>&1
