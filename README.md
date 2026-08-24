# FedHyDRA

End-to-end core-method research code for **Federated Learning with Dual-Scale Hybrid Divergence and Relation-Aware Embedding for Structured Non-IID Data**.

FedHyDRA combines:

1. Laplace-smoothed label histograms and RBF random Fourier feature (RFF) means.
2. An adaptive mixture of Jensen-Shannon divergence and RFF-MMD.
3. A weighted top-k client graph and a variational graph autoencoder (VGAE).
4. Full-covariance GMM soft memberships.
5. Membership-weighted aggregation under explicit partial participation.

This repository is a clean reimplementation from the supplied manuscript, not a claim that the unpublished original experiment source was recovered. The manuscript leaves several implementation details unspecified; every completion is configurable and recorded in each run's `implementation_manifest.json`.

## Features

- CIFAR-100, Tiny-ImageNet, STL-10, and a synthetic CI dataset.
- Fixed-budget IID, Dirichlet (`alpha=0.30`), and structured overlapping partitions.
- FedHyDRA, FedAvg, and FedProx training paths.
- No PyTorch Geometric dependency: the small client graph uses a dense two-layer VGAE.
- Deterministic partition, initialization, and participation schedules.
- Checkpoint/resume, CSV metrics, saved partition manifests, and balanced accuracy.
- Unit tests for the paper equations, graph invariants, aggregation identity, partitioning, VGAE, and a complete smoke run.

The separately published baselines FedGCD, KL-FedDis, FedWaD, FedAF, and FedDNA are not copied into this repository. Their revisions and experiment hyperparameters are not specified by the manuscript, and some do not provide reusable licensed source. The included FedAvg/FedProx controls exercise the common simulator fairly; external baselines should be integrated from their authoritative implementations.

Supported experiment scope:

| Experiment path | Status |
|---|---|
| FedHyDRA core training/evaluation | Included |
| CIFAR-100 / Tiny-ImageNet / STL-10 | Included |
| Structured and random Dirichlet splits | Included |
| FedAvg / FedProx controls | Included |
| Fixed/JSD-only/MMD-only hybrid ablations | Included |
| Spectral no-VGAE and hard-GMM ablations | Included, with explicit completion choices |
| Beta, k, and cadence sensitivity | Available through `--set` overrides |
| Five-seed launch and result aggregation | Included |
| External paper baselines | Not vendored |
| Turnover / temporal-shift / SVHN stress scenarios | Not included; manuscript details are insufficient for exact reconstruction |
| Local-only / Centralized references | Not included |

## Installation

Python 3.10 or newer is required.

```bash
cd FedHyDRA
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a CPU-only environment, install PyTorch from its CPU wheel index before installing this project.

## Quick verification

```bash
pytest -q
fedhydra train --config configs/smoke.yaml --run-dir runs/smoke-check
```

The smoke configuration uses four synthetic clients and exercises the full client-summary, JSD/MMD, graph, VGAE, GMM, aggregation, evaluation, and checkpoint pipeline.

## Main experiments

Structured CIFAR-100:

```bash
fedhydra train --config configs/cifar100.yaml
```

Random non-IID CIFAR-100:

```bash
fedhydra train --config configs/cifar100-random.yaml
```

FedProx control using the exact same partition and participation seeds:

```bash
fedhydra train --config configs/fedprox-cifar100.yaml
```

Ablation examples:

```bash
fedhydra train --config configs/ablation-fixed-hybrid.yaml
fedhydra train --config configs/ablation-jsd-only.yaml
fedhydra train --config configs/ablation-mmd-only.yaml
fedhydra train --config configs/ablation-no-vgae.yaml
fedhydra train --config configs/ablation-hard-gmm.yaml
```

The no-VGAE path uses a deterministic normalized-adjacency spectral embedding before the same GMM. The manuscript does not define its no-VGAE replacement, so this is labeled an explicit ablation completion rather than an exact recovery.

Run another repetition without copying a config:

```bash
fedhydra train --config configs/cifar100.yaml \
  --set experiment.seed=1 \
  --set experiment.name=cifar100-structured-fedhydra
```

Aggregate finished repetitions:

```bash
python scripts/summarize_runs.py \
  --root runs \
  --experiment cifar100-structured-fedhydra \
  --output runs/cifar100-structured-summary.json
```

Inspect a partition without training:

```bash
fedhydra inspect-partition --config configs/cifar100.yaml
```

Resume a run:

```bash
fedhydra train --config configs/cifar100.yaml \
  --resume runs/cifar100-structured-fedhydra/seed-0/checkpoints/round-0020.pt
```

## Dataset layout

### CIFAR-100 and STL-10

The configured torchvision loaders download these datasets when `data.download: true`.

### Tiny-ImageNet

Download and extract `tiny-imagenet-200` first, then point `data.root` at the extracted directory:

```text
data/tiny-imagenet-200/
├── train/
│   └── n01443537/
│       └── images/
└── val/
    ├── images/
    └── val_annotations.txt
```

The validation loader reads the original flat validation layout; no file reorganization is required. An optional downloader is provided:

```bash
python scripts/download_tiny_imagenet.py --destination data
```

## Configuration

YAML files support inheritance through `_base_` and command-line overrides through repeated `--set section.key=value` arguments. The manuscript defaults are represented directly:

| Setting | Value |
|---|---:|
| clients / participants | 200 / 20 |
| rounds / local epochs / batch | 300 / 5 / 32 |
| SGD | lr 0.01, momentum 0.9, weight decay 1e-4, cosine decay |
| RFF dimension | 256 |
| VGAE latent dimension | 16 |
| GMM components | 5 |
| graph neighbors | 20 |
| histogram beta | 1 |
| graph temperature | 0.5 |
| hybrid regularizer | 0.01 |
| refresh intervals | 5 / 10 / 20 |

The MobileNet width, RFF gamma and feature normalization, VGAE architecture/epochs/KL scaling, GMM regularization, local sample budgets, preprocessing, and optimizer learning rates are implementation choices because the manuscript does not report them. See [Implementation notes](docs/IMPLEMENTATION_NOTES.md).

## Partial-participation aggregation

There is a consequential ambiguity in the manuscript. If Eqs. (20)-(23) use the same client set and epsilon is zero, then

```text
sum_k pi_k * cluster_update_k = mean_i(client_update_i),
```

so the memberships cancel exactly. The paper also uses 20 of 200 clients per round but does not define the inactive-client terms in Eq. (22).

This implementation exposes both interpretations:

- `mixture_share_scope: all_clients` (default): `pi_k` comes from all cached clients, while each cluster update uses current participants. This makes relation-aware partial participation operational.
- `mixture_share_scope: participants`: both quantities use participants. This is the literal same-set formula and reduces to a uniform participant mean when epsilon is zero and sample weighting is off.

The identity is locked down in `tests/test_aggregation.py`; it is not hidden by the implementation.

## Structured split status

The paper's CIFAR-100 table uses background/object/animal factors that are not executable CIFAR-100 annotations, and it does not publish client index manifests. Tiny-ImageNet and STL-10 are described only as analogous templates. The included builder therefore supplies a deterministic, configuration-driven benchmark completion:

- disjoint anchor class blocks for persistent groups;
- adjacent overlap masses `[0.20, 0.20, 0.30, 0.10]`;
- seeded boundary clients with fractional group membership;
- a mild, deterministic group-domain color transform to induce feature shift;
- fixed per-client sample budgets without replacement.

Every exact sample index, profile, domain membership, and boundary client is saved to `partition.json`. To reproduce a private/original split, replace `structured_profiles()` or load its indices into the `Partition` structure. See [Structured partitions](docs/STRUCTURED_PARTITIONS.md).

## Outputs

Each run writes:

```text
runs/<experiment>/seed-<seed>/
├── config.json
├── implementation_manifest.json
├── partition.json
├── metrics.csv
├── summary.json
└── checkpoints/
```

`summary.json` reports final balanced accuracy, convergence to 90% of final accuracy, CoV, Min-Mean ratio, and average measured round time. `round_times.csv` records every round even when evaluation is less frequent. Accuracy fields are stored both as fractions and explicit percentage columns. Stability is calculated over all evaluated rounds. The manuscript does not state a burn-in/window, so this choice is explicitly recorded rather than inferred.

The round timer includes local training, summary computation, structural refresh, and aggregation. It excludes evaluation and does not simulate network latency. Structured runs evaluate the shared model on the clean canonical test split; the engineered client-domain shifts are training-side heterogeneity, not shifted test domains.

## Repository layout

```text
src/fedhydra/
├── data/          # dataset adapters, fixed-budget partitioning, domain shifts
├── evaluation/    # balanced accuracy and curve metrics
├── federated/     # client, server, checkpoints, training loop
├── methods/       # summaries, relations, VGAE, GMM, aggregation
└── models/        # lightweight MobileNetV2
configs/           # experiment configurations
docs/              # assumptions and reproduction guidance
scripts/           # utility launchers
tests/             # equation and end-to-end tests
```

## Reported manuscript results

The manuscript reports structured balanced top-1 accuracy of 89.0% on CIFAR-100, 88.6% on Tiny-ImageNet, and 89.5% on STL-10. These numbers are reference targets, not results generated or certified by this repository. Exact table reproduction requires the unpublished original structured partitions and omitted hyperparameters.

## Citation

```bibtex
@unpublished{shin2026fedhydra,
  title   = {Federated Learning with Dual-Scale Hybrid Divergence and
             Relation-Aware Embedding for Structured Non-IID Data},
  author  = {Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Shin, Jitae},
  note    = {Manuscript},
  year    = {2026}
}
```

## License

This reimplementation is released under the MIT License. Dataset licenses and third-party dependencies remain governed by their respective terms.
