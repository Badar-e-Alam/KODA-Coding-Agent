# Benchmarking KODA on Terminal-Bench

Measures KODA's harness — prompt, tools, middleware, context policy — against
Terminal-Bench 2.1 (89 tasks) so that changes to it can be judged by something
other than intuition.

## Why this is worth doing

The Terminal-Bench 2.0 leaderboard carries 142 entries, and the same model
lands in very different places depending on the harness wrapped around it.
GLM 5 under Terminus 2 scores **52.4% ±2.6**. That is the number to beat with a
GLM-5.x model: exceeding it is a harness result, attributable to work done here
rather than to the model underneath.

For reference, the top of the 2.0 board is ~84.7% (GPT-5.5), and GLM 5.2's
published 81.0 / 0.827 is on **2.1** — which is why 2.1 is the dataset wired up
here. Numbers are not comparable across dataset versions.

## Setup

```bash
uv tool install harbor          # the official TB2 harness
open -a Docker                  # tasks run in containers
```

Provider keys must be in a `.env` that Harbor loads — it forwards
`OLLAMA_API_KEY` and friends from its own environment into the task container.
Point `KODA_BENCH_ENV_FILE` at it if it is not the repo's `.env`.

## Running

```bash
bench/run.sh dev                     # 45-task iteration split
bench/run.sh test                    # 44-task held-out split
bench/run.sh all                     # all 89, leaderboard-comparable
bench/run.sh dev -k 3 -n 8           # 3 attempts/task, 8 concurrent

.venv/bin/python bench/report.py bench/jobs/koda-dev --failures
.venv/bin/python bench/report.py bench/jobs/koda-dev --compare bench/jobs/<prior>
```

## How it hangs together

Harbor ships a `langgraph` agent that stages a project directory into the task
container, `uv pip install -e .`s it, and calls
`graph.ainvoke({"messages": [task instruction]}, config)` with the model from
`-m` arriving as `configurable.model`. That is already `build_agent`'s shape, so
there is no custom adapter to write or keep in sync — `bench/graph.py` is a
factory that returns **the same graph the KODA TUI drives**. That matters: the
thing under measurement has to be the real harness, or the number means nothing.

`bench/` is a slim staging root — `koda/` and `coding_agent/` are symlinks that
materialize on copy, so Harbor stages ~1MB per trial instead of the 1.1G repo.

## The two things that differ in a container, and why

Both are opt-in parameters on `build_agent`; the product defaults are untouched.

**`interrupt_on={}`** — KODA gates mutating tools through deepagents'
`HumanInTheLoopMiddleware`, which calls LangGraph's `interrupt()`. Nothing in a
container can answer it. The failure is silent rather than loud: the graph
*stops* and checkpoints, `ainvoke` returns a two-message state carrying
`__interrupt__`, and the trial scores 0 with no exception to explain it. This
cost one confusing run before it was found.

**`virtual_mode=False`** — KODA normally jails its file tools to the project
root, so a model-supplied `/etc/hosts` resolves under `<cwd>/etc/hosts`. TB
tasks address `/etc`, `/usr/local` and `/app` by absolute path, and under the
jail the agent is simply told those files do not exist. See
`tests/test_bench_backend.py`. Note the jail was never a security boundary —
`execute` runs unrestricted shell either way.

## Methodology

**Iterate on dev, not on the full set.** Tuning a harness against all 89 tasks
is fitting the benchmark, and the resulting leaderboard number is not a
measurement of anything. Read failures only from dev. `bench/splits/{dev,test}.txt`
are checked in so the split cannot drift between runs — they alternate through
the alphabetically sorted task list, which deals each task family (all the
`build-*` tasks, say) evenly into both halves instead of stacking them on one
side the way a random draw can.

**Respect the noise floor.** One task on the dev split is 2.2 points. The
leaderboard's own error bars are ±2–3 points on the *full* set. `report.py`
prints a Wilson interval next to every mean for this reason: a 4-point
improvement on a single dev run is not yet evidence of anything. Raise `-k` for
more attempts per task before believing a small delta.

**Pair outcome with process.** Accuracy says whether a change worked; it does
not say why. KODA's Langfuse instrumentation gives the mechanism-level view —
single-tool-turn rate, `think` usage, parallel fan-out — with a recorded
baseline from 2026-06/07 (85.2% of turns fired exactly one tool call). Moving
accuracy without moving those is usually a sign of noise.

## The loop — `bench/rci/`

```bash
.venv/bin/python bench/rci/loop.py --iterations 3
.venv/bin/python bench/rci/loop.py --iterations 1 --tasks 6   # cheap dry run
.venv/bin/python bench/rci/loop.py --report-only              # regenerate REPORT.md
```

Each iteration stacks the next hypothesis onto the accepted harness, runs the
dev split, compares task-by-task against the incumbent, and records the verdict.
The recursion is the accept step: a change that survives becomes part of the
harness the *next* hypothesis is tested against, so improvements compound rather
than each being measured against the original baseline. The accepted stack *is*
the harness; `bench/rci/ledger.jsonl` is its derivation, and `REPORT.md` is that
ledger rendered as "what improved what".

A harness change is data, not a source edit — see `bench/variant.py`. Variants
are written to `bench/variant.json`, which is staged into every trial container,
so any result can be traced back to the exact configuration that produced it.

**Deciding whether a change worked** is the part that determines whether this
loop produces a better harness or just a longer ledger. Comparing means is not
enough: on a 45-task split each task is worth 2.2 points, so flipping two
coin-toss tasks "improves" the score by 4.4 points while doing nothing, and
picking the winner each iteration reliably manufactures an impressive curve over
a harness that never improved. Because both runs cover the *same* tasks, the
results are paired, and the informative quantity is the tasks whose outcome
*changed* — fixed 5 / broke 0 is strong evidence at a 11-point delta, while
fixed 8 / broke 7 is nothing at a comparable one. `bench/rci/stats.py` uses an
exact McNemar test on those discordant pairs, at α=0.10 (loose on purpose — this
is a search, and the held-out split is the real gate).

Rejections are kept in the ledger. Knowing that a plausible change did nothing
is most of what this loop produces, and a record of only the wins is one that
cannot be audited and will re-propose the same failures.

Knobs, roughly in order of expected payoff: `system_prompt_v2.py` blocks ·
`nudges.py` thresholds · tool surface and descriptions · context/compaction
policy · step budget · subagent fan-out policy.

## Prompt corrections, which are not tuning

`bench/prompt_fixes.py` rewrites two prompt blocks before any variant applies.
This is the prompt half of `virtual_mode=False`: KODA's `<Paths>` block tells the
model that "the leading `/` is the project root, NOT the OS root" and that
OS-absolute paths fail — which, once the jail is off, is simply false. Every TB
task statement uses real absolute paths (`/app/data.txt`), so leaving it in place
means grading the model on how well it copes with being misinformed about its own
tools. The blanket "never search from the OS filesystem root" rule goes the same
way, for the same reason.

Both substitutions assert that they matched. A silently-skipped correction would
leave the contradiction in place and invalidate every number produced afterward
— invisible in a results table, which is exactly why it raises.
