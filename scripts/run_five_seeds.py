"""Launch the manuscript's five partition seeds sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cifar100.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for seed in range(5):
        command = [
            sys.executable,
            "-m",
            "fedhydra",
            "train",
            "--config",
            str(args.config),
            "--set",
            f"experiment.seed={seed}",
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

