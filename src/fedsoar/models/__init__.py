"""Model APIs shared with legacy FedHyDRA imports."""

import sys

from fedhydra.models import MobileNetV2Small, TinyConvNet, build_model, mobilenet

sys.modules[__name__ + ".mobilenet"] = mobilenet

__all__ = ["MobileNetV2Small", "TinyConvNet", "build_model"]
