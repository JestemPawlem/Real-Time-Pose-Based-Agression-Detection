from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'RWF-2000'
SPLITS_DIR = DATA_DIR / 'splits'
SPLIT_MANIFEST_PATH = SPLITS_DIR / 'rwf2000_splits.csv'

RANDOM_SEED: int = 42
