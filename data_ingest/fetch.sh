#!/usr/bin/env bash
# Fetch CREMA-D. CSVs come from raw.githubusercontent; the WAVs live in git-lfs
# so raw. serves pointer stubs for those and we sparse-clone the GitLab mirror
# instead. Everything lands in data/raw and data/audio, both gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
GH=https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master

mkdir -p "$RAW/processedResults"
for f in finishedResponses.csv finishedEmoResponses.csv \
         finishedResponsesWithRepeatWithPractice.csv SentenceFilenames.csv \
         VideoDemographics.csv README.md \
         processedResults/summaryTable.csv processedResults/tabulatedVotes.csv; do
  [ -s "$RAW/$f" ] && continue
  echo "fetch $f"
  curl -fsSL --retry 3 -o "$RAW/$f" "$GH/$f"
done

# ~600MB of 16kHz mono WAV. Only needed by modeling/, not by agreement/.
if [ "${1:-}" = "--audio" ]; then
  if [ ! -d data/audio/repo/AudioWAV ]; then
    mkdir -p data/audio
    GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout --depth 1 \
      https://gitlab.com/cs-cooper-lab/crema-d-mirror.git data/audio/repo
    git -C data/audio/repo sparse-checkout init --cone
    git -C data/audio/repo sparse-checkout set AudioWAV
    git -C data/audio/repo checkout
  fi
  git -C data/audio/repo lfs pull --include="AudioWAV/*"
  echo "wav files: $(ls data/audio/repo/AudioWAV | wc -l)"
fi
