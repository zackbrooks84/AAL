"""Test configuration for ensuring package import paths.

This module adjusts ``sys.path`` during test collection so that the
repository root is available for imports without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
