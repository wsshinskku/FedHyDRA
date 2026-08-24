"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fedhydra.config import load_config
from fedhydra.data import build_datasets, build_partition
from fedhydra.federated import FederatedTrainer
from fedhydra.utils import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedhydra")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run a federated experiment")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--resume", type=Path)

    inspect = subparsers.add_parser("inspect-partition", help="summarize a partition")
    inspect.add_argument("--config", required=True, type=Path)
    inspect.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    config = load_config(args.config, args.set)
    if args.command == "train":
        trainer = FederatedTrainer(config, run_dir=args.run_dir, resume=args.resume)
        print(json.dumps(trainer.run(), indent=2))
        return 0
    if args.command == "inspect-partition":
        bundle = build_datasets(config.data, config.experiment.seed)
        partition = build_partition(bundle.train_targets, config.data, config.experiment.seed)
        sizes = np.asarray([len(indices) for indices in partition.client_indices])
        report = {
            "metadata": partition.metadata,
            "client_size_min": int(sizes.min()),
            "client_size_max": int(sizes.max()),
            "assigned_unique_samples": int(
                np.unique(np.concatenate(partition.client_indices)).size
            ),
            "domain_membership_shape": list(partition.domain_memberships.shape),
        }
        print(json.dumps(report, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

