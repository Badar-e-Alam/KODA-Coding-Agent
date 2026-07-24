"""Deciding whether a harness change actually did anything.

The naive comparison — run the variant, see if the mean went up — is how a
tuning loop talks itself into a harness made of noise. On a 45-task split each
task is worth 2.2 points, so a change that flips two coin-toss tasks the right
way "improves" the score by 4.4 points while doing nothing at all. Over a dozen
iterations, picking the winner each time reliably produces an impressive-looking
curve and a harness that is no better than where it started.

Two things guard against that here.

**Pair the comparison.** Variant and baseline run the *same* tasks, so the
results are paired, and the interesting quantity is not two independent
proportions but the tasks whose outcome *changed*. A variant that fixes 5 tasks
and breaks 0 is strong evidence even though the mean only moved 11 points; a
variant that fixes 8 and breaks 7 is nothing, despite also moving the mean.
McNemar's test looks exactly at those discordant pairs, and it is far more
powerful at this N than comparing two independent means — which is what makes
per-iteration decisions affordable at all.

**Report the uncertainty next to every number.** Wilson intervals for the
accuracy itself, an exact p-value for the change. Neither is a licence to stop
thinking, but both make it harder to not notice when a result is thin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Preferred over the normal approximation because this project spends much of
    its time near 0 or 1, where the normal approximation produces intervals that
    run off the end of [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Exact two-sided McNemar p-value.

    Args:
        fixed: Tasks the baseline failed and the variant passed.
        broken: Tasks the baseline passed and the variant failed.

    Under the null hypothesis that the change does nothing, each discordant task
    is an independent coin flip, so the count of "fixed" is Binomial(n, 0.5).
    Uses the exact binomial rather than the chi-square approximation because the
    discordant counts here are small — often single digits — which is precisely
    where the approximation misleads.

    Concordant tasks (both pass, or both fail) carry no information about the
    change and are correctly ignored.
    """
    n = fixed + broken
    if n == 0:
        return 1.0
    # Two-sided: sum the probability of every outcome at least as lopsided.
    observed = _binom_pmf(fixed, n)
    return min(1.0, sum(_binom_pmf(k, n) for k in range(n + 1)
                        if _binom_pmf(k, n) <= observed + 1e-12))


@dataclass
class Comparison:
    """The verdict on one variant, versus whatever it was tested against."""

    n: int
    base_passed: int
    variant_passed: int
    fixed: list[str]
    broken: list[str]
    p_value: float
    accepted: bool
    reason: str

    @property
    def delta(self) -> float:
        return (self.variant_passed - self.base_passed) / self.n if self.n else 0.0


def compare(
    base: dict[str, bool],
    variant: dict[str, bool],
    alpha: float = 0.10,
) -> Comparison:
    """Compare two runs task-by-task and decide whether to keep the change.

    Args:
        base: task name → passed, for the incumbent harness.
        variant: task name → passed, for the candidate.
        alpha: significance threshold. Deliberately looser than 0.05: this is a
            search over candidates, and the held-out split is the real gate. A
            strict threshold here mostly buys the loop the ability to reject
            everything and conclude the harness is unimprovable.

    Only tasks present in both runs are compared, so a crashed or skipped trial
    on one side cannot masquerade as a regression on the other.
    """
    shared = sorted(set(base) & set(variant))
    fixed = [t for t in shared if not base[t] and variant[t]]
    broken = [t for t in shared if base[t] and not variant[t]]
    p = mcnemar_exact(len(fixed), len(broken))

    if not shared:
        accepted, reason = False, "no overlapping tasks between the two runs"
    elif len(fixed) <= len(broken):
        accepted, reason = False, f"fixed {len(fixed)}, broke {len(broken)} — not an improvement"
    elif p > alpha:
        accepted = False
        reason = (f"fixed {len(fixed)}, broke {len(broken)}, but p={p:.3f} > {alpha} "
                  "— within what chance produces at this N")
    else:
        accepted = True
        reason = f"fixed {len(fixed)}, broke {len(broken)}, p={p:.3f} — keeping it"

    return Comparison(
        n=len(shared),
        base_passed=sum(1 for t in shared if base[t]),
        variant_passed=sum(1 for t in shared if variant[t]),
        fixed=fixed,
        broken=broken,
        p_value=p,
        accepted=accepted,
        reason=reason,
    )
