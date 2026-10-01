"""Legacy import namespace for FedSOAR (formerly FedHyDRA)."""

from fedhydra.config import Config, FedHyDRAConfig, FedSOARConfig, load_config

__all__ = ["Config", "FedHyDRAConfig", "FedSOARConfig", "load_config"]
__version__ = "0.2.0"

