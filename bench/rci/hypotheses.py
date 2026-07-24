"""Candidate harness changes, in the order the loop will try them.

Each entry is a hypothesis with a stated reason to believe it, not a guess.
They come from two places: the mismatch between what KODA's prompt assumes (a
long-lived session inside a user's git repo) and what Terminal-Bench tasks are
(one-shot, self-contained, judged by a script), plus the Langfuse trace analysis
that found the agent firing exactly one tool call in 85% of turns.

Ordering is roughly by expected effect per unit of benchmark time, because each
of these costs a full dev-split run to evaluate. Cheap, high-prior changes go
first so that if the loop is stopped early, the runs already spent were the ones
most likely to matter.

Two of these are controls that could plausibly make things *worse*
(`nudges-off`, in particular). That is deliberate: a loop that only proposes
changes it expects to win has no way to discover that an existing mechanism is
hurting, and KODA's nudge middleware has never been measured against an outcome
— only against the process metric it was designed to move.

Every hypothesis uses `prompt_append` or `env` rather than `prompt_edits`.
Appends cannot fail to apply, so a hypothesis can never silently degrade into
"baseline under a different name" the way a stale find/replace anchor can.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from variant import Variant  # noqa: E402

HYPOTHESES: list[Variant] = [
    Variant(
        name="terminal-framing",
        rationale=(
            "KODA's prompt is written for an ongoing session inside a user's "
            "codebase — it talks about projects, git, and the user. A "
            "Terminal-Bench task is a single self-contained job on a machine the "
            "agent owns, graded by a script it never sees, with no user to ask. "
            "The mismatch costs turns on irrelevant orientation and, worse, "
            "invites the agent to stop and ask when it should decide."
        ),
        prompt_append=(
            "<Task context>\n"
            "You are solving ONE self-contained task on a machine you have to "
            "yourself, non-interactively. There is no user to ask and no session "
            "after this one: `ask_user` cannot be answered, so never wait on it — "
            "decide, act, and state the assumption you made.\n\n"
            "The work is judged by an automated check against the task statement, "
            "run after you stop. Nothing you say is graded; only the state you "
            "leave the filesystem in. So spend your turns changing that state and "
            "proving it correct, not narrating or summarizing.\n"
            "</Task context>"
        ),
    ),
    Variant(
        name="literal-requirements",
        rationale=(
            "The <Verify> block prescribes codebase verification — write a pytest "
            "file, run it, delete it. Terminal-Bench tasks are graded against the "
            "literal words of the task statement (exact paths, exact output "
            "format, exact file names), and the most common way an otherwise "
            "correct solution fails is missing one stated requirement rather than "
            "getting the hard part wrong. Re-reading the statement as a checklist "
            "targets that failure directly."
        ),
        prompt_append=(
            "<Before you finish>\n"
            "Re-read the task statement one final time and treat every concrete "
            "noun in it as a requirement to check: each exact path, each file "
            "name, each output format, each named command. Confirm each one "
            "against the real filesystem with `execute` — not from memory of what "
            "you intended to do.\n\n"
            "A solution that is 95% right and misses one stated requirement scores "
            "exactly the same as one that was never attempted. Check the boring "
            "requirements as carefully as the hard one.\n"
            "</Before you finish>"
        ),
    ),
    Variant(
        name="no-knowledge-base",
        rationale=(
            "The <Project knowledge base> block instructs the agent to maintain "
            "AGENTS.md and topic pages. In a throwaway container graded by a test "
            "script, every one of those writes is a wasted turn — and the block is "
            "one of the longest in the prompt, so it also competes for attention "
            "with instructions that do affect the score."
        ),
        prompt_append=(
            "<Task context addendum>\n"
            "This container is discarded when you stop. Do not create or update "
            "AGENTS.md, user_preferences.md, project_history.md, or any other "
            "knowledge-base page — none of it will ever be read. Write only files "
            "the task itself asks for.\n"
            "</Task context addendum>"
        ),
    ),
    Variant(
        name="nudge-threshold-2",
        rationale=(
            "The serial-read nudge fires after 3 consecutive single-read turns. "
            "Traces showed streaks of 7–9, so by the time it fires several "
            "round-trips are already spent. Firing at 2 corrects sooner; the risk "
            "is that it interrupts legitimately sequential exploration, which is "
            "what the measurement is for."
        ),
        env={"KODA_NUDGE_THRESHOLD": "2"},
    ),
    Variant(
        name="nudges-off",
        rationale=(
            "Control. The nudge middleware was built to move a process metric "
            "(single-tool-turn rate) and has never been tested against an outcome. "
            "It injects text into context mid-task, which costs tokens and may "
            "derail a working line of reasoning. If turning it off does not hurt, "
            "it is not earning its complexity; if it helps, that is a finding."
        ),
        env={"KODA_DISABLE_NUDGES": "1"},
    ),
]
