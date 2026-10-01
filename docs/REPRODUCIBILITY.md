# Reproducibility guide

## Five-seed experiments

```bash
python scripts/run_five_seeds.py --config configs/cifar100.yaml
python scripts/summarize_runs.py --root runs --experiment cifar100-structured-fedsoar --output runs/cifar100-structured-summary.json
```

The launcher runs seeds 0–4 sequentially under one experiment name. Individual seeds can also be selected with `--set experiment.seed=0`.

For comparisons, change `federated.method` and method-specific settings while keeping dataset and training settings fixed. Use a separate experiment name for each method to preserve its output directory. Partition generation, initialization, and client participation each derive from the experiment seed.

## Randomness and environment

- Python, NumPy, CPU PyTorch, and CUDA generators are seeded.
- cuDNN benchmarking is disabled.
- Deterministic algorithms are requested with warnings for unsupported kernels.
- DataLoader workers derive their seeds from PyTorch.
- Client participation uses `seed + 202`, RFF uses `seed + 101`, and GMM uses `seed + 303`.

Record package versions and hardware because floating-point behavior depends on the PyTorch/CUDA/cuDNN environment.

## Reporting results

Preserve `config.json`, `implementation_manifest.json`, `partition.json`, and the generated metric files. Report:

- aggregation scope (`all_clients` or `participants`), sample weighting, and epsilon;
- the stability metric window;
- GPU, CPU, memory, and package versions;
- seed count, mean, sample standard deviation, and confidence intervals.

Run `pytest -q` to verify the implementation before changing the experiment configuration.

## Manuscript comparison

Use the revised FedSOAR manuscript's Table 5 only as a reported reference; it is not a cached output from this repository. Main profiles match the stated 200-client, 20-participant, 300-round, five-local-epoch protocol and the 5/10/20 refresh cadences. The repository supplies a deterministic structural surrogate, not the unpublished original sample-index assignments. Read [manuscript alignment](MANUSCRIPT_ALIGNMENT.md) before interpreting a run as a reproduction.

For the Section 5.5 sensitivity sweeps, use the same seed set and one distinct experiment name per setting:

```bash
fedsoar train --config configs/cifar100.yaml --set fedsoar.histogram_beta=0.1 --set experiment.name=cifar100-beta-0.1
fedsoar train --config configs/cifar100.yaml --set fedsoar.graph_neighbors=10 --set experiment.name=cifar100-k-10
fedsoar train --config configs/cifar100.yaml --set fedsoar.hybrid_interval=2 --set fedsoar.embedding_interval=5 --set fedsoar.clustering_interval=10 --set experiment.name=cifar100-cadence-2-5-10
```

The manuscript varies beta over {0.01, 0.1, 1, 10}, k over {5, 10, 20, 40}, and cadences over {(2,5,10), (5,10,20), (10,20,40), (20,40,80)}. Run all five seeds per setting before comparing means; do not infer the manuscript's reported gains from the CPU smoke profile.
