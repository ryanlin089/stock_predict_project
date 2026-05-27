from pathlib import Path
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import keras

from services.constants import CHECKPOINT_DIR, DAYS_TO_LOOK_BACK, MODEL_TYPES
from services.errors import ErrorType, StockPredictionError
from services.model_wrappers import KerasWrapper
from services.stock_service import missing_checkpoint_files

logger = logging.getLogger(__name__)

model_cache: dict[str, dict] = {}


def load_models_for_stock(code: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> dict:
    if code in model_cache:
        return model_cache[code]

    missing_files = missing_checkpoint_files(code, checkpoint_dir)
    if missing_files:
        missing_names = ", ".join(path.name for path in missing_files)
        raise StockPredictionError(
            ErrorType.MODEL_FILE_NOT_FOUND,
            f"模型權重不完整，缺少：{missing_names}",
        )

    try:
        selected_features = joblib.load(checkpoint_dir / f"{code}_features.joblib")
        if not isinstance(selected_features, list) or len(selected_features) == 0:
            raise ValueError("features 檔案格式錯誤或為空")

        loaded = {
            "features": selected_features,
            "models": {},
            "scalers": {},
        }

        for model_type in MODEL_TYPES:
            scaler_path = checkpoint_dir / f"{code}_{model_type}_scaler.joblib"
            loaded["scalers"][model_type] = joblib.load(scaler_path)

            if model_type == "xgb":
                loaded["models"][model_type] = joblib.load(checkpoint_dir / f"{code}_xgb.joblib")
            else:
                keras_model = keras.models.load_model(checkpoint_dir / f"{code}_{model_type}.keras")
                loaded["models"][model_type] = KerasWrapper(
                    keras_model=keras_model,
                    window_size=DAYS_TO_LOOK_BACK,
                    num_features=len(selected_features),
                )

        model_cache[code] = loaded
        return loaded

    except StockPredictionError:
        raise
    except Exception as exc:
        logger.exception("Failed to load model for stock %s", code)
        raise StockPredictionError(
            ErrorType.MODEL_LOAD_FAILED,
            f"載入 {code} 模型失敗：{exc}",
        ) from exc
