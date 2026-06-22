from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


APP_TITLE = "Анализ оттока клиентов"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_processed.parquet"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "catboost_churn"
)


MODEL_PATH = ARTIFACT_DIR / "model.cbm"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
INPUT_TEMPLATE_PATH = ARTIFACT_DIR / "input_template.csv"