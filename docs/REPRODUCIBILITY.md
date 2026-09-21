# Reproducibility guide

## Five-seed experiments

```bash
python scripts/run_five_seeds.py --config configs/cifar100.yaml
python scripts/summarize_runs.py --root runs --experiment cifar100-structured-fedhydra --output runs/cifar100-structured-summary.json
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
