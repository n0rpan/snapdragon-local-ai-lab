"""pytest: isolate HOME/XDG for the whole suite before any test module is imported."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402

_isolation.isolate()
