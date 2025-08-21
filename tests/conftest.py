import sys
from pathlib import Path

# Ensure the local package is imported instead of any installed site-packages
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# If a different 'aic' is already imported, drop it so tests use the local one
sys.modules.pop("aic", None)
