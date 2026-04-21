from pathlib import Path

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RAW_TEXT_PATH = RAW_DATA_DIR / "input.txt"
VOCAB_PATH = PROCESSED_DATA_DIR / "vocab.json"
TRAIN_IDS_PATH = PROCESSED_DATA_DIR / "train_ids.pt"
VAL_IDS_PATH = PROCESSED_DATA_DIR / "val_ids.pt"

# =========================
# Data config
# =========================
TRAIN_RATIO = 0.9
BLOCK_SIZE = 64
BATCH_SIZE = 16

# =========================
# Model config
# =========================
N_EMBD = 128
DROPOUT = 0.1

# =========================
# Debug / runtime
# =========================
SEED = 42
DEVICE = "cuda"