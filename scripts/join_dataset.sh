#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if command -v shasum >/dev/null 2>&1; then
  CHECK=(shasum -a 256)
else
  CHECK=(sha256sum)
fi
"${CHECK[@]}" -c PARTS_SHA256SUMS
cat twinreactor-material-boron-5000.zip.part-* > twinreactor-material-boron-5000.zip
"${CHECK[@]}" -c SHA256SUMS
printf '\nVerified dataset archive: %s/twinreactor-material-boron-5000.zip\n' "$ROOT"
printf 'Extract it with: unzip twinreactor-material-boron-5000.zip\n'
