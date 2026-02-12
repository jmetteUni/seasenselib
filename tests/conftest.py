"""
Test configuration to ensure local source is imported.

Pytest can otherwise resolve an installed `seasenselib` package from
site-packages. We force the repository root to the front of sys.path
to guarantee tests execute against the working tree.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
