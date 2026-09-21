# FedHyDRA

**Official implementation** of **Federated Learning with Dual-Scale Hybrid Divergence and Relation-Aware Embedding for Structured Non-IID Data**.

[English](README.md) | [한국어](README.ko.md)

**Authors:** Wooseok Shin, Janghoon Yang, Zhiqiang Shen, and Jitae Shin.

FedHyDRA learns client relationships from label and feature distributions to aggregate federated updates under structured non-IID data.

## Method

1. Clients compute Laplace-smoothed label histograms and shared random Fourier feature (RFF) means.
2. The server adaptively combines Jensen–Shannon divergence and RFF-MMD.
3. A weighted client graph and a two-layer variational graph autoencoder (VGAE) produce relation-aware embeddings.
4. A full-covariance Gaussian mixture model estimates soft memberships.
5. The server aggregates active-client updates using cached population mixture shares and periodically refreshes the hybrid weight, embeddings, and clusters.

The package includes **FedHyDRA, FedAvg, and FedProx**, plus fixed-hybrid, JSD-only, MMD-only, no-VGAE, and hard-membership ablations. The no-VGAE path uses normalized-adjacency spectral embedding.

## Installation

Requirements: **Python 3.10+**, PyTorch, torchvision, NumPy, scikit-learn, PyYAML, and tqdm. The smoke profile runs on CPU; image-dataset experiments support CUDA.

```bash
git clone https://github.com/wsshinskku/FedHyDRA.git
cd FedHyDRA
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```bash
fedhydra train --config configs/smoke.yaml --run-dir runs/smoke-check
pytest -q
```

The synthetic smoke profile uses four clients, two participants per round, and two rounds. It exercises local training, summary extraction, graph learning, clustering, aggregation, evaluation, and checkpointing.

## Datasets and experiments

| Dataset | Structured partition | Dirichlet partition |
|---|---|---|
| CIFAR-100 | [cifar100.yaml](configs/cifar100.yaml) | [cifar100-random.yaml](configs/cifar100-random.yaml) |
| Tiny-ImageNet | [tiny-imagenet.yaml](configs/tiny-imagenet.yaml) | [tiny-imagenet-random.yaml](configs/tiny-imagenet-random.yaml) |
| STL-10 | [stl10.yaml](configs/stl10.yaml) | [stl10-random.yaml](configs/stl10-random.yaml) |

CIFAR-100 and STL-10 download through torchvision when `data.download: true`. Prepare Tiny-ImageNet with:

```bash
python scripts/download_tiny_imagenet.py --destination data
```

The Tiny-ImageNet loader reads the standard `train/<class>/images`, `val/images`, and `val_annotations.txt` layout. Set dataset storage with `data.root`.

```bash
fedhydra train --config configs/cifar100.yaml
fedhydra inspect-partition --config configs/cifar100.yaml
fedhydra train --config configs/cifar100.yaml --set experiment.seed=1
fedhydra train --config configs/cifar100.yaml --set federated.method=fedavg --set experiment.name=cifar100-structured-fedavg
fedhydra train --config configs/fedprox-cifar100.yaml
```

The CIFAR-100 profile uses 200 clients, 20 participants per round, 300 rounds, five local epochs, and MobileNetV2 with width multiplier 0.5. Structured partitions combine adjacent class-group overlap, boundary clients, and label-preserving color shifts. Exact sample indices and memberships are saved in `partition.json`.

Ablation profiles: [fixed hybrid](configs/ablation-fixed-hybrid.yaml), [JSD only](configs/ablation-jsd-only.yaml), [MMD only](configs/ablation-mmd-only.yaml), [no VGAE](configs/ablation-no-vgae.yaml), and [hard memberships](configs/ablation-hard-gmm.yaml).

## Multiple seeds and checkpoints

```bash
python scripts/run_five_seeds.py --config configs/cifar100.yaml
python scripts/summarize_runs.py --root runs --experiment cifar100-structured-fedhydra --output runs/cifar100-structured-summary.json
fedhydra train --config configs/cifar100.yaml --resume runs/cifar100-structured-fedhydra/seed-0/checkpoints/round-0020.pt
```

The seed script runs seeds 0–4. The summarizer writes JSON and Markdown with means, sample standard deviations, and normal-approximation 95% confidence intervals.

## Outputs

Default path: `runs/<experiment>/seed-<seed>/`. Use `--run-dir` to choose an explicit location.

| Artifact | Contents |
|---|---|
| `config.json` | Effective experiment settings |
| `implementation_manifest.json` | Package versions, device, model size, and algorithm settings |
| `partition.json` | Exact sample indices and structured memberships |
| `metrics.csv` | Evaluation metrics and server diagnostics |
| `round_times.csv` | Time for each communication round |
| `summary.json` | Balanced accuracy, convergence, stability, and mean round time |
| `checkpoints/` | Training state for resuming runs |

Balanced top-1 accuracy averages per-class recall. Convergence is the first evaluated round reaching 90% of final accuracy. CoV and Min/Mean use all evaluation points. Round timing includes local training, summaries, server refresh, and aggregation; evaluation occurs outside that window. Structured color shifts apply to client training data, and evaluation uses the clean test split.

## Repository guide

| Path | Purpose |
|---|---|
| [src/fedhydra](src/fedhydra) | Data, models, client/server training, methods, and metrics |
| [configs](configs) | Dataset profiles, controls, and ablations |
| [scripts](scripts) | Dataset preparation, multi-seed execution, and reporting |
| [tests](tests) | Mathematical, data, checkpoint, and integration checks |
| [Implementation notes](docs/IMPLEMENTATION_NOTES.md) | Equations, update schedules, and aggregation |
| [Structured partitions](docs/STRUCTURED_PARTITIONS.md) | Partition generation and custom partition API |
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Seed protocol and reporting settings |

## Citation

```bibtex
@unpublished{shin2026fedhydra,
  title  = {Federated Learning with Dual-Scale Hybrid Divergence and Relation-Aware Embedding for Structured Non-IID Data},
  author = {Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Shin, Jitae},
  note   = {Manuscript},
  year   = {2026}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata. Source code is distributed under the [MIT License](LICENSE).
