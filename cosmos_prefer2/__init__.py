# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos-Prefer2.5: Unified environment for Cosmos Predict2.5 and Transfer2.5.

This package provides a unified working environment with access to both
cosmos-predict2.5 and cosmos-transfer2.5 World Foundation Models.
"""

__version__ = "0.1.0"

import os
import sys

# Auto-configure PYTHONPATH when imported
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PREDICT_DIR = os.path.join(_PARENT_DIR, "cosmos-predict2.5")
_TRANSFER_DIR = os.path.join(_PARENT_DIR, "cosmos-transfer2.5")

# Add paths if they exist and aren't already in sys.path
for _path in [_PREDICT_DIR, _TRANSFER_DIR]:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# Clean up namespace
del os, sys, _PARENT_DIR, _PREDICT_DIR, _TRANSFER_DIR, _path

