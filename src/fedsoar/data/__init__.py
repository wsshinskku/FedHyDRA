"""Dataset and partition APIs shared with legacy FedHyDRA imports."""

import sys
from importlib import import_module

from fedhydra.data import ClientDataset, DatasetBundle, Partition, build_datasets, build_partition

for _module in ("datasets", "partition"):
    globals()[_module] = import_module(f"fedhydra.data.{_module}")
    sys.modules[f"{__name__}.{_module}"] = globals()[_module]

__all__ = ["ClientDataset", "DatasetBundle", "Partition", "build_datasets", "build_partition"]
