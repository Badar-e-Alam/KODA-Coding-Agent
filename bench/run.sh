#!/usr/bin/env bash
#
# Run KODA against a Terminal-Bench 2.1 split via Harbor.
#
#   bench/run.sh dev                 # the 45-task iteration split
#   bench/run.sh test                # the 44-task held-out split — see below
#   bench/run.sh all                 # all 89, for a leaderboard-comparable number
#   bench/run.sh dev -k 3 -n 8       # extra args pass straight through to harbor
#
# The dev/test split exists so that tuning the harness does not quietly become
# tuning *to the benchmark*. Iterate on dev. Read failures only from dev. Touch
# test rarely — every look at it spends a little of its power to tell you
# whether a gain is real, and that budget does not refill.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${1:?usage: bench/run.sh <dev|test|all> [extra harbor args...]}"
shift || true

DATASET="${KODA_BENCH_DATASET:-terminal-bench/terminal-bench-2-1}"
MODEL="${KODA_BENCH_MODEL:-ollama/glm-5.2}"
JOBS_DIR="${KODA_BENCH_JOBS_DIR:-$REPO/bench/jobs}"
# Harbor forwards OLLAMA_API_KEY (and the other provider keys) from its own
# environment into the task container, so the keys have to be loaded here.
ENV_FILE="${KODA_BENCH_ENV_FILE:-$REPO/.env}"

if ! command -v harbor >/dev/null 2>&1; then
  echo "harbor not on PATH — install with: uv tool install harbor" >&2
  exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "env file not found: $ENV_FILE" >&2
  echo "point KODA_BENCH_ENV_FILE at a .env containing OLLAMA_API_KEY" >&2
  exit 1
fi

# Build the task filter. `all` passes no -i at all so harbor runs the dataset
# whole; a split is expanded into one -i per task name.
filters=()
if [ "$SPLIT" != "all" ]; then
  split_file="$REPO/bench/splits/$SPLIT.txt"
  [ -f "$split_file" ] || { echo "unknown split: $SPLIT (expected dev|test|all)" >&2; exit 1; }
  while IFS= read -r task; do
    [ -n "$task" ] && filters+=(-i "$task")
  done < "$split_file"
  echo "split '$SPLIT': ${#filters[@]} filters → $(( ${#filters[@]} / 2 )) tasks"
fi

exec harbor run \
  -d "$DATASET" \
  -a langgraph \
  --ak "project_path=$REPO/bench" \
  --ak graph=koda \
  -m "$MODEL" \
  --env-file "$ENV_FILE" \
  -o "$JOBS_DIR" \
  --job-name "koda-$SPLIT" \
  -y \
  ${filters[@]+"${filters[@]}"} \
  "$@"
