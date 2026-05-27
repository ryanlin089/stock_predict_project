import argparse
import logging
from pathlib import Path
import traceback

import joblib

from services.constants import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DAYS_TO_LOOK_BACK,
    TRAIN_TEST_SPLIT,
    SUPPORTED_STOCKS,
)
from services.preprocessing import data_preprocessing, get_selected_features
from services.stock_service import check_model_files_exist
from services.training import (
    prepare_data_and_train_xgb,
    prepare_data_and_train_rnn,
    prepare_data_and_train_lstm,
    prepare_data_and_train_transformer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_one_stock(code: str, force: bool = False):
    name = SUPPORTED_STOCKS[code]

    if check_model_files_exist(code) and not force:
        logger.info("[%s %s] 已有完整新版 checkpoint，跳過訓練。使用 --force 可重新訓練。", code, name)
        return

    logger.info("[%s %s] 開始資料前處理", code, name)
    train_csv_path, test_csv_path = data_preprocessing(code, DATA_DIR)

    logger.info("[%s %s] 開始選擇 features", code, name)
    selected_features = get_selected_features(train_csv_path)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_features, CHECKPOINT_DIR / f"{code}_features.joblib")
    logger.info("[%s %s] features: %s", code, name, selected_features)

    logger.info("[%s %s] 開始訓練 XGBoost", code, name)
    prepare_data_and_train_xgb(
        file_path=train_csv_path,
        selected_features=selected_features,
        window_size=DAYS_TO_LOOK_BACK,
        test_split=TRAIN_TEST_SPLIT,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    logger.info("[%s %s] 開始訓練 RNN", code, name)
    prepare_data_and_train_rnn(
        file_path=train_csv_path,
        selected_features=selected_features,
        window_size=DAYS_TO_LOOK_BACK,
        test_split=TRAIN_TEST_SPLIT,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    logger.info("[%s %s] 開始訓練 LSTM", code, name)
    prepare_data_and_train_lstm(
        file_path=train_csv_path,
        selected_features=selected_features,
        window_size=DAYS_TO_LOOK_BACK,
        test_split=TRAIN_TEST_SPLIT,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    logger.info("[%s %s] 開始訓練 Transformer", code, name)
    prepare_data_and_train_transformer(
        file_path=train_csv_path,
        selected_features=selected_features,
        window_size=DAYS_TO_LOOK_BACK,
        test_split=TRAIN_TEST_SPLIT,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    logger.info("[%s %s] 訓練完成。train=%s, test=%s", code, name, train_csv_path.name, test_csv_path.name)


def main():
    parser = argparse.ArgumentParser(description="Batch train all supported stock models")
    parser.add_argument("--force", action="store_true", help="重新訓練並覆蓋既有 checkpoints")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("資料資料夾：%s", DATA_DIR)
    logger.info("模型資料夾：%s", CHECKPOINT_DIR)

    for code in SUPPORTED_STOCKS:
        try:
            train_one_stock(code, force=args.force)
        except Exception as exc:
            logger.error("[%s %s] 訓練失敗：%s", code, SUPPORTED_STOCKS[code], exc)
            logger.error(traceback.format_exc())
            continue

    logger.info("批次訓練流程結束")


if __name__ == "__main__":
    main()
