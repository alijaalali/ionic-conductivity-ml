"""Shared paths. Everything is relative to this repository checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                  # derived feature table (master_table.parquet)
OBELIX_DATA = DATA_DIR / "obelix"         # OBELiX files (CC BY 4.0), see README.md
RESULTS = ROOT / "results"                # JSON/CSV outputs written by the analysis scripts
FIG_DIR = ROOT / "figures"                # figure files written by the plotting scripts
