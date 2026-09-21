"""End-to-end deterministic simulator for FedAvg, FedProx, and FedHyDRA."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fedhydra.config import Config
from fedhydra.data import ClientDataset, build_datasets, build_partition
from fedhydra.evaluation import evaluate_model, summarize_curve
from fedhydra.federated.client import ClientResult, train_client
from fedhydra.federated.server import FedHyDRAServer
from fedhydra.methods.aggregation import aggregate_fedavg, apply_delta
from fedhydra.methods.summaries import RandomFourierFeatures, compute_client_summary
from fedhydra.models import build_model
from fedhydra.utils import (
    LOGGER,
    atomic_json_dump,
    count_parameters,
    resolve_device,
    seed_everything,
    worker_seed,
)


class FederatedTrainer:
    def __init__(
        self,
        config: Config,
        run_dir: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.experiment.seed, config.experiment.deterministic)
        self.device = resolve_device(config.experiment.device)
        default_run = (
            Path(config.experiment.output_dir)
            / config.experiment.name
            / f"seed-{config.experiment.seed}"
        )
        self.run_dir = Path(run_dir) if run_dir is not None else default_run
        if resume is None and self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"run directory is not empty: {self.run_dir}. "
                "Choose --run-dir or resume an existing checkpoint."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.bundle = build_datasets(config.data, config.experiment.seed)
        self.partition = build_partition(
            self.bundle.train_targets, config.data, config.experiment.seed
        )
        self.model = build_model(
            config.model, self.bundle.num_classes, self.bundle.channels
        ).to(self.device)
        self.rff = RandomFourierFeatures(
            input_dimension=self.model.feature_dimension,
            output_dimension=config.fedhydra.rff_dimension,
            gamma=config.fedhydra.rff_gamma,
            seed=config.experiment.seed + 101,
        )
        self.histogram_cache = np.full(
            (config.data.num_clients, self.bundle.num_classes),
            1.0 / self.bundle.num_classes,
            dtype=np.float64,
        )
        self.rff_cache = np.zeros(
            (config.data.num_clients, config.fedhydra.rff_dimension), dtype=np.float64
        )
        self.summary_valid = np.zeros(config.data.num_clients, dtype=bool)
        self.last_seen = np.full(config.data.num_clients, -1, dtype=np.int64)
        self.server = None
        if config.federated.method == "fedhydra":
            self.server = FedHyDRAServer(
                config.fedhydra,
                clients=config.data.num_clients,
                node_feature_dimension=self.bundle.num_classes
                + config.fedhydra.rff_dimension,
                device=self.device,
                seed=config.experiment.seed + 303,
            )
        self.schedule = self._participation_schedule()
        self.metrics: list[dict[str, Any]] = []
        self.round_times: list[dict[str, Any]] = []
        self.start_round = 0
        self._write_manifest()
        if resume is not None:
            self._load_checkpoint(Path(resume))

    def _write_manifest(self) -> None:
        packages = {}
        for package in ("numpy", "PyYAML", "scikit-learn", "torch", "torchvision", "tqdm"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        atomic_json_dump(self.config.to_dict(), self.run_dir / "config.json")
        atomic_json_dump(self.partition.to_jsonable(), self.run_dir / "partition.json")
        atomic_json_dump(
            {
                "implementation": "Official FedHyDRA implementation",
                "parameter_count": count_parameters(self.model),
                "device": str(self.device),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": packages,
                    "cuda_runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "gpu_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                },
                "aggregation_scope": self.config.fedhydra.mixture_share_scope,
                "hybrid_mode": self.config.fedhydra.hybrid_mode,
                "embedding_mode": self.config.fedhydra.embedding_mode,
                "hard_memberships": self.config.fedhydra.hard_memberships,
                "structured_partition_template": self.partition.metadata.get("template"),
                "notes": [
                    "Structured partitions use seeded adjacent class anchors and boundary-client mixtures.",
                    "Model architecture, RFF bandwidth, and VGAE training settings are saved in config.json.",
                    "all_clients uses population mixture shares and participant-only cluster means.",
                    "Structured-domain shifts are applied to training clients; evaluation uses the clean test split.",
                    "Round timing excludes evaluation and does not simulate network latency.",
                ],
            },
            self.run_dir / "implementation_manifest.json",
        )

    def _participation_schedule(self) -> np.ndarray:
        rng = np.random.default_rng(self.config.experiment.seed + 202)
        return np.stack(
            [
                np.sort(
                    rng.choice(
                        self.config.data.num_clients,
                        size=self.config.data.clients_per_round,
                        replace=False,
                    )
                )
                for _ in range(self.config.federated.rounds)
            ]
        )

    def _client_dataset(self, client_id: int, summary: bool = False) -> ClientDataset:
        transform = (
            self.bundle.summary_transform if summary else self.bundle.train_transform
        )
        memberships = None
        shift = 0.0
        if self.config.data.partition == "structured":
            memberships = self.partition.domain_memberships[client_id]
            shift = self.config.data.domain_shift_strength
        return ClientDataset(
            self.bundle.train,
            self.partition.client_indices[client_id],
            transform,
            memberships,
            shift,
        )

    def _summary_loader(self, client_id: int) -> DataLoader:
        workers = self.config.data.num_workers
        return DataLoader(
            self._client_dataset(client_id, summary=True),
            batch_size=self.config.federated.summary_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_seed,
        )

    def _update_summary(self, client_id: int, round_index: int) -> None:
        summary = compute_client_summary(
            model=self.model,
            loader=self._summary_loader(client_id),
            rff=self.rff,
            num_classes=self.bundle.num_classes,
            beta=self.config.fedhydra.histogram_beta,
            device=self.device,
            normalize_features=self.config.fedhydra.normalize_features,
            max_samples=self.config.federated.max_summary_samples,
        )
        self.histogram_cache[client_id] = summary.label_histogram
        self.rff_cache[client_id] = summary.rff_mean
        self.summary_valid[client_id] = True
        self.last_seen[client_id] = round_index

    def bootstrap_summaries(self) -> None:
        missing = np.flatnonzero(~self.summary_valid)
        if not missing.size:
            return
        LOGGER.info("Bootstrapping summaries for %d clients", missing.size)
        for client_id in tqdm(missing.tolist(), desc="summary bootstrap", leave=False):
            self._update_summary(client_id, round_index=-1)

    def _test_loader(self) -> DataLoader:
        return DataLoader(
            self.bundle.test,
            batch_size=max(self.config.federated.batch_size, 128),
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_seed,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def _train_participants(self, round_index: int, participants: np.ndarray) -> list[ClientResult]:
        results: list[ClientResult] = []
        for client_id in tqdm(participants.tolist(), desc=f"round {round_index}", leave=False):
            if self.server is not None:
                self._update_summary(client_id, round_index)
            result = train_client(
                client_id=client_id,
                global_model=self.model,
                dataset=self._client_dataset(client_id, summary=False),
                config=self.config.federated,
                device=self.device,
                round_index=round_index,
                total_rounds=self.config.federated.rounds,
                seed=self.config.experiment.seed + round_index * 100_003 + client_id,
                num_workers=self.config.data.num_workers,
            )
            results.append(result)
        return results

    def _aggregate(self, results: list[ClientResult]) -> tuple[dict[str, torch.Tensor], np.ndarray | None]:
        updates = [result.update for result in results]
        counts = [result.sample_count for result in results]
        participants = [result.client_id for result in results]
        if self.server is None:
            return aggregate_fedavg(updates, counts), None
        return self.server.aggregate(updates, participants, counts)

    def _append_metric(self, row: dict[str, Any]) -> None:
        self.metrics.append(row)
        path = self.run_dir / "metrics.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _append_timing(self, row: dict[str, Any]) -> None:
        self.round_times.append(row)
        path = self.run_dir / "round_times.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _rewrite_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.unlink(missing_ok=True)
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _checkpoint(self, next_round: int) -> Path:
        path = self.run_dir / "checkpoints" / f"round-{next_round:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        payload = {
            "next_round": next_round,
            "model": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
            "histogram_cache": torch.from_numpy(self.histogram_cache),
            "rff_cache": torch.from_numpy(self.rff_cache),
            "summary_valid": torch.from_numpy(self.summary_valid),
            "last_seen": torch.from_numpy(self.last_seen),
            "metrics": self.metrics,
            "round_times": self.round_times,
            "server": None if self.server is None else self.server.state_dict(),
            "config": self.config.to_dict(),
            "rng": {
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("config") != self.config.to_dict():
            raise ValueError("checkpoint configuration differs from the requested configuration")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.histogram_cache = checkpoint["histogram_cache"].numpy()
        self.rff_cache = checkpoint["rff_cache"].numpy()
        self.summary_valid = checkpoint["summary_valid"].numpy()
        self.last_seen = checkpoint["last_seen"].numpy()
        self.metrics = list(checkpoint["metrics"])
        self.round_times = list(checkpoint.get("round_times", []))
        self.start_round = int(checkpoint["next_round"])
        if self.server is not None and checkpoint["server"] is not None:
            self.server.load_state_dict(checkpoint["server"])
        rng = checkpoint.get("rng")
        if rng is not None:
            torch.random.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        self._rewrite_csv(self.run_dir / "metrics.csv", self.metrics)
        self._rewrite_csv(self.run_dir / "round_times.csv", self.round_times)
        LOGGER.info("Resumed from %s at round %d", path, self.start_round)

    def run(self) -> dict[str, Any]:
        if self.server is not None and self.config.federated.bootstrap_summaries:
            self.bootstrap_summaries()
        elif self.server is not None and not self.summary_valid.all():
            raise ValueError(
                "FedHyDRA requires complete summary caches at the first structural refresh; "
                "enable federated.bootstrap_summaries"
            )

        test_loader = self._test_loader()
        last_checkpoint: Path | None = None
        for round_index in range(self.start_round, self.config.federated.rounds):
            participants = self.schedule[round_index]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            results = self._train_participants(round_index, participants)
            refresh: Any = None
            if self.server is not None:
                refresh = self.server.refresh(
                    round_index, self.histogram_cache, self.rff_cache
                )
            delta, priors = self._aggregate(results)
            updated = apply_delta(
                self.model.state_dict(), delta, self.config.federated.server_learning_rate
            )
            self.model.load_state_dict(updated)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            self._append_timing(
                {
                    "round": round_index + 1,
                    "round_seconds": elapsed,
                    "participants": json.dumps(participants.tolist()),
                }
            )

            evaluated = round_index % self.config.federated.eval_every == 0
            if evaluated or round_index == self.config.federated.rounds - 1:
                evaluation = evaluate_model(
                    self.model, test_loader, self.bundle.num_classes, self.device
                )
                row = {
                    "round": round_index + 1,
                    "balanced_accuracy": evaluation.balanced_accuracy,
                    "balanced_accuracy_percent": 100.0 * evaluation.balanced_accuracy,
                    "accuracy": evaluation.accuracy,
                    "accuracy_percent": 100.0 * evaluation.accuracy,
                    "test_loss": evaluation.loss,
                    "client_loss": float(np.mean([result.mean_loss for result in results])),
                    "round_seconds": elapsed,
                    "omega": None if self.server is None else self.server.omega,
                    "wss": None if self.server is None else self.server.wss,
                    "omega_updated": None if refresh is None else refresh.omega_updated,
                    "embedding_updated": None if refresh is None else refresh.embedding_updated,
                    "clustering_updated": None if refresh is None else refresh.clustering_updated,
                    "mixture_priors": None
                    if priors is None
                    else json.dumps(priors.tolist()),
                    "participants": json.dumps(participants.tolist()),
                }
                self._append_metric(row)
                LOGGER.info(
                    "round=%d balanced_acc=%.4f loss=%.4f time=%.2fs omega=%s",
                    round_index + 1,
                    evaluation.balanced_accuracy,
                    evaluation.loss,
                    elapsed,
                    "-" if self.server is None else f"{self.server.omega:.3f}",
                )
            if (round_index + 1) % self.config.federated.checkpoint_every == 0:
                last_checkpoint = self._checkpoint(round_index + 1)

        if last_checkpoint is None or self.config.federated.rounds % self.config.federated.checkpoint_every:
            last_checkpoint = self._checkpoint(self.config.federated.rounds)
        curve = summarize_curve(
            [float(row["balanced_accuracy"]) for row in self.metrics],
            [int(row["round"]) for row in self.metrics],
            burn_in=0,
        )
        summary = {
            "experiment": self.config.experiment.name,
            "seed": self.config.experiment.seed,
            "method": self.config.federated.method,
            "curve": curve,
            "mean_round_seconds": float(
                np.mean([float(row["round_seconds"]) for row in self.round_times])
            ),
            "last_checkpoint": str(last_checkpoint),
            "metric_window_note": "All evaluated rounds, with zero burn-in.",
            "accuracy_unit": "fraction in curve; metrics.csv also includes explicit percent columns",
            "timing_note": "All rounds; includes training/summary/server work, excludes evaluation and network simulation.",
        }
        atomic_json_dump(summary, self.run_dir / "summary.json")
        return summary
