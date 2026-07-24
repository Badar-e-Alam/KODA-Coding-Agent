"""Corrections that make the prompt describe the harness the bench actually runs.

These are not tuning. They are the prompt half of the two container deltas in
``graph.py``: once ``virtual_mode=False`` makes the file tools address the real
filesystem, any prompt text still describing a project-rooted virtual filesystem
is simply *false*, and the agent is being lied to about its own tools.

Measuring a harness whose instructions contradict its behavior does not measure
KODA — it measures how well a model copes with being misinformed. So these are
applied to the baseline, before any variant, and the RCI loop tunes on top of a
harness that is at least internally consistent.

Each substitution asserts that it matched. A silently-skipped correction would
leave the contradiction in place and quietly invalidate every number produced
afterward, which is exactly the kind of failure that is invisible in a results
table.
"""

from __future__ import annotations

import re

# What the agent is told about paths once the jail is off. Mirrors the real
# semantics of build_backend(virtual_mode=False): absolute paths are OS paths,
# relative paths resolve against cwd.
_BENCH_PATHS_BLOCK = """<Paths>
File paths in tool calls are **real OS paths**. A leading `/` is the filesystem
root, exactly as it is in the shell.

- Right: `/app/data.txt`, `/etc/nginx/nginx.conf`, `/usr/local/bin/tool`
- Also fine: `data.txt` (relative — resolved against your current directory).
- The task statement's paths are literal. If it says `/app/decomp.c`, pass
  exactly `/app/decomp.c`; do not rewrite or re-root it.

Read and write anywhere on the filesystem the task requires. `read_file`,
`write_file`, `edit_file`, `glob` and `grep` all address real paths, and agree
with what `execute` sees.
</Paths>"""

# The project-scoping rule is right for a coding agent inside someone's repo and
# wrong here: these tasks live in /app, /etc and /usr/local, and the agent has
# the machine to itself. The cost-control half of the rule is worth keeping —
# walking all of / is still slow — so only the prohibition is rewritten.
_BENCH_SEARCH_RULE = (
    "- **Search where the task lives.** You have the whole machine; the work is "
    "usually under `/app`, but may be anywhere the task names. Scope searches to "
    "the directory that plausibly holds the answer rather than walking `/` — not "
    "because other paths are off-limits, but because a full-disk walk is slow and "
    "floods your context. Bound output (`| head`, `wc -l`, `-q`) and pass a "
    "`timeout` for anything slow."
)


def _sub_block(prompt: str, tag: str, replacement: str) -> str:
    pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL)
    new, count = pattern.subn(lambda _: replacement, prompt, count=1)
    if count != 1:
        raise ValueError(
            f"bench prompt correction failed: no <{tag}> block found. "
            "The prompt has changed; re-check bench/prompt_fixes.py before "
            "trusting any result produced with it."
        )
    return new


def align_prompt_with_bench_semantics(prompt: str) -> str:
    """Rewrite the prompt so it matches how the tools actually behave here."""
    prompt = _sub_block(prompt, "Paths", _BENCH_PATHS_BLOCK)

    # The "never search from /" bullet. Anchored on its distinctive opening so a
    # reworded tail does not silently skip the correction.
    search_rule = re.compile(
        r"^- \*\*Never search from the OS filesystem root\.\*\*.*$", re.MULTILINE
    )
    prompt, count = search_rule.subn(lambda _: _BENCH_SEARCH_RULE, prompt, count=1)
    if count != 1:
        raise ValueError(
            "bench prompt correction failed: the filesystem-root search rule was "
            "not found. Re-check bench/prompt_fixes.py."
        )
    return prompt
