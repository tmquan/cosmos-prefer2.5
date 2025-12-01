#!/usr/bin/env python3
"""Test script to verify the cosmos-prefer2.5 environment setup."""

import sys
import os

def main():
    print("=" * 60)
    print("Cosmos-Prefer2.5 Environment Test")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check PYTHONPATH
    print("PYTHONPATH entries:")
    for path in sys.path[:10]:
        print(f"  - {path}")
    print()
    
    # Test imports
    print("Testing imports:")
    
    tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
    ]
    
    for module, name in tests:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
    
    # Test cosmos packages
    print()
    print("Testing Cosmos packages:")
    
    cosmos_tests = [
        ("cosmos_predict2", "Cosmos Predict2"),
        ("cosmos_transfer2", "Cosmos Transfer2"),
        ("cosmos_oss", "Cosmos OSS"),
    ]
    
    for module, name in cosmos_tests:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
    
    # Test CUDA
    print()
    print("CUDA status:")
    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"  Error checking CUDA: {e}")
    
    print()
    print("=" * 60)
    print("Environment test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
