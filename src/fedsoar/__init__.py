"""FedSOAR public API with backwards-compatible FedHyDRA implementations."""

import sys

from fedhydra import Config, FedHyDRAConfig, FedSOARConfig, __version__, config, load_config

# Share module/class identities with legacy imports and serialized user objects.
sys.modules[__name__ + ".config"] = config

__all__ = ["Config", "FedHyDRAConfig", "FedSOARConfig", "__version__", "load_config"]
