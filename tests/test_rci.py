"""The recursive harness-improvement machinery in bench/rci/.

The loop's job is to decide whether a harness change helped. The failure mode
worth testing against is not a crash — it is the loop confidently accepting
noise, which produces a plausible-looking improvement curve and a harness that
is no better than where it started.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "bench" / "rci"))

from prompt_fixes import align_prompt_with_bench_semantics  # noqa: E402
from stats import compare, mcnemar_exact, wilson  # noqa: E402
from variant import Variant  # noqa: E402


# ── decision statistics ─────────────────────────────────────────────────


def test_symmetric_changes_are_not_evidence() -> None:
    """Fixing 8 and breaking 7 moves the mean but proves nothing."""
    assert mcnemar_exact(8, 7) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_lopsided_changes_are_evidence() -> None:
    assert mcnemar_exact(6, 0) < 0.05
    assert mcnemar_exact(5, 0) < 0.10


def test_noise_is_rejected() -> None:
    """A variant that swaps which tasks pass must not be accepted."""
    base = {f"t{i}": i < 20 for i in range(45)}
    noisy = dict(base)
    noisy["t20"], noisy["t21"] = True, True   # two newly passing
    noisy["t0"], noisy["t1"] = False, False   # two newly failing

    verdict = compare(base, noisy)
    assert not verdict.accepted
    assert verdict.delta == 0.0


def test_real_improvement_is_accepted() -> None:
    base = {f"t{i}": i < 20 for i in range(45)}
    better = {f"t{i}": i < 26 for i in range(45)}

    verdict = compare(base, better)
    assert verdict.accepted
    assert len(verdict.fixed) == 6
    assert not verdict.broken


def test_small_gains_stay_rejected_at_this_n() -> None:
    """Two extra passes out of 45 is not a finding, however tempting."""
    base = {f"t{i}": i < 20 for i in range(45)}
    slightly = {**base, "t20": True, "t21": True}

    assert not compare(base, slightly).accepted


def test_only_shared_tasks_are_compared() -> None:
    """A trial missing from one run must not read as a regression."""
    base = {"a": True, "b": True, "c": False}
    variant = {"a": True, "c": True}  # 'b' never ran

    verdict = compare(base, variant)
    assert verdict.n == 2
    assert verdict.fixed == ["c"]
    assert not verdict.broken


def test_wilson_interval_stays_in_bounds() -> None:
    for successes, n in ((0, 45), (45, 45), (1, 3)):
        lo, hi = wilson(successes, n)
        assert 0.0 <= lo <= hi <= 1.0


# ── variants ────────────────────────────────────────────────────────────


def test_accepted_variants_compose() -> None:
    """The recursion: an accepted change joins the harness under test."""
    a = Variant(name="a", prompt_append="AAA", env={"X": "1"})
    b = Variant(name="b", prompt_append="BBB", env={"Y": "2"})

    stacked = a.stack(b, name="a+b", rationale="")
    prompt = stacked.apply_prompt("BASE")

    assert "AAA" in prompt and "BBB" in prompt
    assert stacked.env == {"X": "1", "Y": "2"}


def test_stale_prompt_anchor_fails_loudly(tmp_path: Path) -> None:
    """A variant that cannot apply must raise, not silently score as baseline."""
    import pytest

    stale = Variant(name="stale", prompt_edits=[("GONE", "x")])
    with pytest.raises(ValueError, match="anchor not found"):
        stale.apply_prompt("a prompt without that text")


def test_variant_round_trips_through_disk(tmp_path: Path) -> None:
    original = Variant(name="v", rationale="why", prompt_append="P", env={"K": "V"})
    original.save(tmp_path)
    loaded = Variant.load(tmp_path)

    assert loaded.name == original.name
    assert loaded.env == original.env
    assert loaded.prompt_append == original.prompt_append


def test_missing_variant_is_baseline(tmp_path: Path) -> None:
    assert Variant.load(tmp_path).is_baseline


def test_corrupt_variant_raises_rather_than_mislabelling(tmp_path: Path) -> None:
    """Falling back to baseline would file the result under the wrong name."""
    import pytest

    (tmp_path / "variant.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable variant"):
        Variant.load(tmp_path)


# ── prompt corrections ──────────────────────────────────────────────────


def test_bench_prompt_describes_real_paths() -> None:
    """With virtual_mode=False, the virtual-root instruction is a lie."""
    from coding_agent.agent import _render_system_prompt

    before = _render_system_prompt(Path("/app"))
    after = align_prompt_with_bench_semantics(before)

    assert "NOT the OS root" in before      # the claim that must go
    assert "NOT the OS root" not in after
    assert "real OS paths" in after
    assert "Never search from the OS filesystem root" not in after


def test_prompt_correction_fails_loudly_when_prompt_changes() -> None:
    """Silently skipping a correction would invalidate every later result."""
    import pytest

    with pytest.raises(ValueError, match="no <Paths> block found"):
        align_prompt_with_bench_semantics("a prompt with no blocks")
