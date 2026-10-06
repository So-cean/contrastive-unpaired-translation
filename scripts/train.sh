#!/usr/bin/env bash
# One portable training job per invocation.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_medical.sh" "$@"
