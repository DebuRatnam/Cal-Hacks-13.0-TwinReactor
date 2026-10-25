#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-$PWD/twinreactor_dataset_download}"
mkdir -p "$DEST"
gh release download dataset-material-boron-5000-v3 \
  --repo DebuRatnam/Cal-Hacks-13.0-TwinReactor \
  --dir "$DEST" --skip-existing \
  --pattern 'twinreactor-material-boron-5000.zip.part-*' \
  --pattern PARTS_SHA256SUMS --pattern SHA256SUMS --pattern join_dataset.sh
bash "$DEST/join_dataset.sh"
printf '\nThe full dataset ZIP is ready in %s\n' "$DEST"
