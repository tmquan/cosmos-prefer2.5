# Cosmos-Prefer2.5
# Educational repository for Cosmos World Foundation Models

__version__ = "0.1.0"
__author__ = "NVIDIA Corporation"
__license__ = "Apache-2.0"

# Import only utilities that don't require cosmos packages
from .utils.environment import setup_pythonpath, check_environment

# Lazy imports for inference pipelines (they require cosmos packages)
def __getattr__(name):
    """Lazy import for pipelines to avoid requiring cosmos packages at import time."""
    if name == "Predict2Pipeline":
        from .inference.predict_pipeline import Predict2Pipeline
        return Predict2Pipeline
    elif name == "Transfer2Pipeline":
        from .inference.transfer_pipeline import Transfer2Pipeline
        return Transfer2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "setup_pythonpath",
    "check_environment",
    "Predict2Pipeline",
    "Transfer2Pipeline",
]

