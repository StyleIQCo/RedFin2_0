import sys
from pathlib import Path

# Make `src` and `data` importable without install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
