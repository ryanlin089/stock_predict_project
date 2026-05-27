import re
from pathlib import Path

from services.constants import CHECKPOINT_DIR, MODEL_TYPES, SUPPORTED_STOCKS


def normalize_code(code: str) -> str:
    return str(code).strip()


def is_valid_code_format(code: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", code))


def get_stock_name(code: str) -> str | None:
    return SUPPORTED_STOCKS.get(code)


def required_checkpoint_files(code: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> list[Path]:
    files: list[Path] = []
    for model_type in MODEL_TYPES:
        if model_type == "xgb":
            files.append(checkpoint_dir / f"{code}_xgb.joblib")
        else:
            files.append(checkpoint_dir / f"{code}_{model_type}.keras")
        files.append(checkpoint_dir / f"{code}_{model_type}_scaler.joblib")

    files.append(checkpoint_dir / f"{code}_features.joblib")
    return files


def missing_checkpoint_files(code: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> list[Path]:
    return [path for path in required_checkpoint_files(code, checkpoint_dir) if not path.exists()]


def check_model_files_exist(code: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> bool:
    return len(missing_checkpoint_files(code, checkpoint_dir)) == 0


def get_trained_stocks() -> list[dict]:
    return [
        {
            "code": code,
            "name": name,
            "is_trained": check_model_files_exist(code),
        }
        for code, name in SUPPORTED_STOCKS.items()
    ]
