"""Inference pipelines for Cosmos models"""

# Lazy imports to avoid requiring cosmos packages at import time
def __getattr__(name):
    """Lazy import for pipelines."""
    if name == "Predict2Pipeline":
        from .predict_pipeline import Predict2Pipeline
        return Predict2Pipeline
    elif name == "Transfer2Pipeline":
        from .transfer_pipeline import Transfer2Pipeline
        return Transfer2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Predict2Pipeline",
    "Transfer2Pipeline",
]

