"""Aggregate completed seed summaries into JSON and a Markdown table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def statistic(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "n": len(array),
        "mean": float(array.mean()),
        "std": standard_deviation,
        "ci95_half_width": float(1.96 * standard_deviation / np.sqrt(len(array))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment", help="optional exact experiment name")
    parser.add_argument("--output", type=Path, help="optional aggregate JSON path")
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.root.rglob("summary.json")):
        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if args.experiment and summary["experiment"] != args.experiment:
            continue
        groups[(summary["experiment"], summary["method"])].append(summary)
    if not groups:
        raise SystemExit("No matching summary.json files found")

    aggregate: dict[str, Any] = {}
    print("| Experiment | Method | Seeds | Balanced accuracy (%) | Convergence | Time/round (s) |")
    print("|---|---|---:|---:|---:|---:|")
    for (experiment, method), runs in sorted(groups.items()):
        final = statistic([100.0 * float(run["curve"]["final"]) for run in runs])
        convergence = statistic(
            [float(run["curve"]["convergence_round_90pct_final"]) for run in runs]
        )
        timing = statistic([float(run["mean_round_seconds"]) for run in runs])
        key = f"{experiment}:{method}"
        aggregate[key] = {
            "experiment": experiment,
            "method": method,
            "seeds": sorted(int(run["seed"]) for run in runs),
            "balanced_accuracy_percent": final,
            "convergence_round": convergence,
            "mean_round_seconds": timing,
        }
        print(
            f"| {experiment} | {method} | {final['n']} | "
            f"{final['mean']:.3f} +/- {final['std']:.3f} | "
            f"{convergence['mean']:.2f} | {timing['mean']:.3f} |"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(aggregate, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

