"""Evaluation APIs shared with legacy FedHyDRA imports."""

import sys

from fedhydra.evaluation import ClassificationMetrics, evaluate_model, metrics, summarize_curve

sys.modules[__name__ + ".metrics"] = metrics

__all__ = ["ClassificationMetrics", "evaluate_model", "summarize_curve"]
