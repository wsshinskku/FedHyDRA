# Reproducibility checklist

## Five manuscript repetitions

Run seeds 0 through 4 using the same experiment name. Bash:

```bash
for seed in 0 1 2 3 4; do
  fedhydra train --config configs/cifar100.yaml --set experiment.seed=$seed
done
```

PowerShell:

```powershell
0..4 | ForEach-Object {
  fedhydra train --config configs/cifar100.yaml --set "experiment.seed=$_"
}
```

Change only `federated.method` and method-specific settings to compare algorithms. The partition, initialization, and participation schedule are all derived independently from the same experiment seed.

## Determinism

- Python, NumPy, CPU torch, and all CUDA generators are seeded.
- cuDNN benchmarking is disabled.
- Deterministic algorithms are requested with warnings for unsupported kernels.
- DataLoader workers derive seeds from torch's worker seed.
- Client participation is precomputed from `seed + 202`.
- RFF uses `seed + 101`; GMM uses `seed + 303`.

Exact bitwise equality across different PyTorch/CUDA/cuDNN versions is not guaranteed. Record package versions and hardware with published results.

## Before reporting results

- Run `pytest -q`.
- Preserve `config.json`, `implementation_manifest.json`, and `partition.json`.
- Report whether aggregation uses `all_clients` or `participants` mixture shares.
- Report the stability metric window.
- Report GPU, CPU, memory, PyTorch, torchvision, CUDA, and scikit-learn versions.
- Distinguish measured results from manuscript reference numbers.

