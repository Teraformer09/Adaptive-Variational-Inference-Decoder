# Minimal standalone conftest for isolated layer branch
import pathlib
import sys
import pytest

_LAYER3_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_LAYER3_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAYER3_ROOT))
