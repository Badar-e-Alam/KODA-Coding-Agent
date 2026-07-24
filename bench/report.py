"""Summarize a Harbor job: accuracy, a confidence interval, and the failures.

    .venv/bin/python bench/report.py bench/jobs/koda-dev
    .venv/bin/python bench/report.py bench/jobs/koda-dev --compare bench/jobs/baseline

The interval is the point of this script. A 45-task dev split moves 2.2 points
per task, so a run that scores 4 points better than the last one has told you
almost nothing on its own. Printing the CI next to the mean makes that visible
at the moment you are deciding whether a harness change worked, which is the
moment it is easiest to fool yourself.

Uses a Wilson score interval rather than the normal approximation: at the
small-N and near-0/near-1 rates this project will spend most of its time in,
the normal approximation produces intervals that run past 0 or 1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default: 95%)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_trials(job_dir: Path) -> list[dict]:
    """Collect per-trial outcomes from a harbor job directory."""
    trials = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        result_path = trial_dir / "result.json"
        if not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError):
            continue
        # Trial dirs are named "<task>__<hash>"; the task name is the stable part.
        task = trial_dir.name.rsplit("__", 1)[0]
        reward = data.get("reward")
        if reward is None:
            metrics = data.get("metrics") or {}
            reward = metrics.get("reward") or metrics.get("mean")
        trials.append(
            {
                "task": task,
                "reward": float(reward) if isinstance(reward, (int, float)) else 0.0,
                "exception": data.get("exception") or data.get("error"),
                "dir": trial_dir,
            }
        )
    return trials


def summarize(job_dir: Path) -> dict:
    trials = load_trials(job_dir)
    n = len(trials)
    passed = sum(1 for t in trials if t["reward"] >= 1.0)
    lo, hi = wilson(passed, n)
    return {
        "job": job_dir.name,
        "n": n,
        "passed": passed,
        "accuracy": passed / n if n else 0.0,
        "ci": (lo, hi),
        "errors": sum(1 for t in trials if t["exception"]),
        "trials": trials,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--compare", type=Path, help="A prior job to diff against")
    ap.add_argument("--failures", action="store_true", help="List failing tasks")
    args = ap.parse_args()

    s = summarize(args.job_dir)
    lo, hi = s["ci"]
    print(f"{s['job']}: {s['passed']}/{s['n']} = {s['accuracy']:.1%}  "
          f"[95% CI {lo:.1%} – {hi:.1%}]  errors={s['errors']}")

    if args.compare:
        b = summarize(args.compare)
        blo, bhi = b["ci"]
        delta = s["accuracy"] - b["accuracy"]
        print(f"{b['job']}: {b['passed']}/{b['n']} = {b['accuracy']:.1%}  "
              f"[95% CI {blo:.1%} – {bhi:.1%}]")
        print(f"\ndelta: {delta:+.1%}")
        # Overlapping intervals are not proof of no effect, but non-overlapping
        # ones are a reasonable bar for "stop debating and keep the change".
        if lo > bhi or hi < blo:
            print("intervals do not overlap — the change is doing something")
        else:
            print("intervals overlap — indistinguishable from noise at this N")

    if args.failures:
        failed = [t for t in s["trials"] if t["reward"] < 1.0]
        print(f"\n{len(failed)} failing tasks:")
        for t in sorted(failed, key=lambda t: t["task"]):
            note = f"  ({t['exception']})" if t["exception"] else ""
            print(f"  {t['task']}{note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
