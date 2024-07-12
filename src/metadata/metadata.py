from pathlib import Path

DATA_DIR = Path(__file__).absolute().parents[2] / "data"
SAVED_MODELS_DIR = DATA_DIR.parent / "saved_models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IDX_TO_LABEL_PATH = RAW_DATA_DIR / "idx_to_label.json"
METADATA_JSON_PATH = RAW_DATA_DIR / "metadata.json"
