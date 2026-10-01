"""FedSOAR structural components (legacy namespace)."""

from fedhydra.methods.aggregation import aggregate_fedavg, aggregate_fedhydra, aggregate_fedsoar
from fedhydra.methods.relations import RelationEngine
from fedhydra.methods.summaries import RandomFourierFeatures

__all__ = [
    "RandomFourierFeatures",
    "RelationEngine",
    "aggregate_fedavg",
    "aggregate_fedhydra",
    "aggregate_fedsoar",
]

