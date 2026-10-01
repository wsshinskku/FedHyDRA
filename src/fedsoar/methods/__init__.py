"""FedSOAR structural components and legacy-compatible aggregation aliases."""

import sys
from importlib import import_module

from fedhydra.methods import (
    RandomFourierFeatures,
    RelationEngine,
    aggregate_fedavg,
    aggregate_fedhydra,
    aggregate_fedsoar,
)

for _module in ("aggregation", "gmm", "relations", "summaries", "vgae"):
    globals()[_module] = import_module(f"fedhydra.methods.{_module}")
    sys.modules[f"{__name__}.{_module}"] = globals()[_module]

__all__ = [
    "RandomFourierFeatures",
    "RelationEngine",
    "aggregate_fedavg",
    "aggregate_fedhydra",
    "aggregate_fedsoar",
]
