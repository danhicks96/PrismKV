#!/usr/bin/env python
"""
run_sidecar.py — Start the PrismKV HTTP compression sidecar.

Usage:
    python scripts/run_sidecar.py [--host HOST] [--port PORT]

Examples:
    python scripts/run_sidecar.py                    # localhost:8765
    python scripts/run_sidecar.py --port 9000        # custom port
    python scripts/run_sidecar.py --host 0.0.0.0     # LAN accessible
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prismkv.sidecar import _main

if __name__ == "__main__":
    _main()
