"""Inference pipelines and helpers for Cosmos models"""

# Lazy imports to avoid requiring cosmos packages at import time
def __getattr__(name):
    """Lazy import for pipelines and helpers."""
    if name == "Predict2Pipeline":
        from .predict_pipeline import Predict2Pipeline
        return Predict2Pipeline
    elif name == "Transfer2Pipeline":
        from .transfer_pipeline import Transfer2Pipeline
        return Transfer2Pipeline
    elif name == "build_inference_pipeline":
        from .predict_helpers import build_inference_pipeline
        return build_inference_pipeline
    elif name == "build_transfer_pipeline":
        from .transfer_helpers import build_transfer_pipeline
        return build_transfer_pipeline
    elif name == "setup_inference_environment":
        from .predict_helpers import setup_inference_environment
        return setup_inference_environment
    elif name == "setup_transfer_environment":
        from .transfer_helpers import setup_transfer_environment
        return setup_transfer_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Pipeline classes
    "Predict2Pipeline",
    "Transfer2Pipeline",
    # Helper functions
    "build_inference_pipeline",
    "build_transfer_pipeline",
    "setup_inference_environment",
    "setup_transfer_environment",
]
