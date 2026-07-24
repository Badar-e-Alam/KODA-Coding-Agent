"""A harness variant: one candidate change to KODA, expressed as data.

The RCI loop tunes the harness by proposing variants, scoring them on
Terminal-Bench, and keeping the ones that clear the noise floor. For that to
work, a "change to the harness" has to be something the loop can write down,
diff, revert, and stack — not an edit to a source file. This module is that
representation.

Variants travel into the task container the simple way: `bench/` is staged into
every trial, so the loop writes `bench/variant.json` before a run and
`bench/graph.py` reads it on the inside. No env-var plumbing through Harbor, and
the exact configuration under test is a file sitting in the staged project of
every trial directory — so a result can always be traced back to what produced
it, months later.

Knobs are deliberately few. Each one maps to something the trace analysis in
`koda-harness-fix-recommendations` identified as load-bearing, and every knob
added is another dimension the search has to cover with expensive runs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VARIANT_FILENAME = "variant.json"


@dataclass
class Variant:
    """One candidate harness configuration.

    Attributes:
        name: Stable identifier, used as the experiment name in the ledger.
        rationale: Why this change might help — the hypothesis being tested.
            Recorded so a result six iterations later still explains itself.
        prompt_edits: ``(find, replace)`` pairs applied to the raw system-prompt
            template, in order. A pair whose ``find`` is absent is an error, not
            a no-op: silently skipping it would mean scoring a variant that
            isn't the one described.
        prompt_append: Text appended to the end of the system prompt.
        env: Environment variables set before the agent is built. Reaches the
            existing switches (``KODA_NUDGE_THRESHOLD``, ``KODA_DISABLE_NUDGES``)
            without new plumbing.
    """

    name: str = "baseline"
    rationale: str = "Unmodified harness."
    prompt_edits: list[tuple[str, str]] = field(default_factory=list)
    prompt_append: str = ""
    env: dict[str, str] = field(default_factory=dict)

    # ── serialization ───────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Variant":
        return cls(
            name=data.get("name", "baseline"),
            rationale=data.get("rationale", ""),
            # JSON has no tuples; normalize the round-trip back to pairs.
            prompt_edits=[tuple(e) for e in data.get("prompt_edits", [])],
            prompt_append=data.get("prompt_append", ""),
            env=dict(data.get("env", {})),
        )

    @classmethod
    def load(cls, directory: Path | None = None) -> "Variant":
        """Read the variant staged alongside this module, or the baseline."""
        directory = directory or Path(__file__).resolve().parent
        path = directory / VARIANT_FILENAME
        if not path.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            # Falling back to baseline would silently mislabel the run — the
            # ledger would record a variant's name against the baseline harness.
            raise RuntimeError(f"unreadable variant at {path}: {exc}") from exc

    def save(self, directory: Path) -> Path:
        path = directory / VARIANT_FILENAME
        path.write_text(self.to_json())
        return path

    # ── application ─────────────────────────────────────────────────────

    def stack(self, other: "Variant", name: str, rationale: str) -> "Variant":
        """Compose this variant with `other` on top.

        This is what makes the loop recursive rather than a flat sweep: once a
        change is accepted it becomes part of the harness that the next
        hypothesis is tested against, so improvements build on each other
        instead of each being measured against the original baseline.
        """
        return Variant(
            name=name,
            rationale=rationale,
            prompt_edits=[*self.prompt_edits, *other.prompt_edits],
            prompt_append=(self.prompt_append + other.prompt_append),
            env={**self.env, **other.env},
        )

    def apply_env(self) -> None:
        for key, value in self.env.items():
            os.environ[key] = str(value)
            logger.info("variant env: %s=%s", key, value)

    def apply_prompt(self, prompt: str) -> str:
        """Apply this variant's edits to a rendered system prompt."""
        for find, replace in self.prompt_edits:
            if find not in prompt:
                raise ValueError(
                    f"variant {self.name!r}: prompt_edits anchor not found: {find[:80]!r}. "
                    "The prompt has changed since this variant was written; "
                    "scoring it would measure a different harness than described."
                )
            prompt = prompt.replace(find, replace, 1)
        if self.prompt_append:
            prompt = prompt.rstrip() + "\n\n" + self.prompt_append.strip() + "\n"
        return prompt

    @property
    def is_baseline(self) -> bool:
        return not (self.prompt_edits or self.prompt_append or self.env)
