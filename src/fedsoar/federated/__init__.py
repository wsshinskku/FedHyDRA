"""Federated training APIs shared with legacy FedHyDRA imports."""

import sys
from importlib import import_module

from fedhydra.federated import FederatedTrainer
from fedhydra.federated.server import FedHyDRAServer, FedSOARServer

for _module in ("client", "server", "trainer"):
    globals()[_module] = import_module(f"fedhydra.federated.{_module}")
    sys.modules[f"{__name__}.{_module}"] = globals()[_module]

__all__ = ["FederatedTrainer", "FedHyDRAServer", "FedSOARServer"]
