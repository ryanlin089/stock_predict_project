from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from services.constants import DATA_DIR, MODEL_TYPES, SUPPORTED_STOCKS, DAYS_TO_LOOK_BACK
from services.errors import ErrorType, StockPredictionError
from services.model_loader import load_models_for_stock

logger = logging.getLogger(__name__)


def test_recent_month_and_predict_tomorrow(
    file_path: str | Path,
    trained_model,
    fitted_scaler,
    selected_features: list[str],
    window_size: int = 3,
) -> tuple[pd.DataFrame, int]:
    path = Path(file_path)
    if not path.exists():
        raise StockPredictionError(
            ErrorType.DATA_FILE_NOT_FOUND,
            f"找不到測試資料：{path}",
        )

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        raise StockPredictionError(
            ErrorType.PREDICTION_FAILED,
            f"測試資料缺少訓練時使用的特徵：{', '.join(missing_features)}",
        )

    last_date = df["Date"].max()
    start_date = last_date - pd.Timedelta(days=30)
    df_recent = df[df["Date"] >= start_date].reset_index(drop=True)

    if len(df_recent) < window_size:
        raise StockPredictionError(
            ErrorType.PREDICTION_FAILED,
            f"近一個月資料不足，至少需要 {window_size} 筆交易日資料",
        )

    scaled_features = fitted_scaler.transform(df_recent[selected_features])
    labels = df_recent["Label"].values
    dates = df_recent["Date"].values

    X_test = []
    y_test = []
    window_dates = []

    for i in range(len(scaled_features) - window_size + 1):
        window_data = scaled_features[i : i + window_size]
        X_test.append(window_data.flatten())
        y_test.append(labels[i + window_size - 1])
        window_dates.append(dates[i + window_size - 1])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    window_dates = np.array(window_dates)

    known_mask = y_test != -1
    unknown_mask = y_test == -1

    X_known = X_test[known_mask]
    y_known = y_test[known_mask]
    X_unknown = X_test[unknown_mask]

    if len(X_known) > 0:
        y_pred_known = trained_model.predict(X_known)
        acc = accuracy_score(y_known, y_pred_known)
        logger.info("Recent month backtest accuracy: %.4f", acc)

    if len(X_unknown) == 0:
        raise StockPredictionError(
            ErrorType.PREDICTION_FAILED,
            "測試資料中沒有 Label = -1 的待預測列，請確認 data preprocessing 是否保留最後一筆資料",
        )

    probabilities = trained_model.predict_proba(X_unknown)[:, 1]
    signal = 0

    # 沿用原始程式邏輯：>= 60% 視為 signal=1；<= 40% 或中間區間視為 signal=0。
    # API 不回傳機率，只回傳上漲 / 下跌。
    for probability in probabilities:
        prob_percent = probability * 100
        if prob_percent >= 60:
            signal = 1
        elif prob_percent <= 40:
            signal = 0
        else:
            signal = 0

    return df_recent, signal


def predict_stock(code: str) -> dict:
    if code not in SUPPORTED_STOCKS:
        raise StockPredictionError(
            ErrorType.UNSUPPORTED_STOCK,
            "目前尚未支援此股票代號",
        )

    test_file_path = DATA_DIR / f"{code}_test.csv"
    if not test_file_path.exists():
        raise StockPredictionError(
            ErrorType.DATA_FILE_NOT_FOUND,
            f"找不到測試資料 {test_file_path.name}，請先執行 python train_all.py 產生 train/test 資料",
        )

    try:
        loaded = load_models_for_stock(code)
        selected_features = loaded["features"]
        models = loaded["models"]
        scalers = loaded["scalers"]

        model_results: dict[str, str] = {}
        signals: dict[str, int] = {}

        for model_type in MODEL_TYPES:
            _, signal = test_recent_month_and_predict_tomorrow(
                file_path=test_file_path,
                trained_model=models[model_type],
                fitted_scaler=scalers[model_type],
                selected_features=selected_features,
                window_size=DAYS_TO_LOOK_BACK,
            )
            signals[model_type] = int(signal)
            model_results[model_type] = "上漲" if signal == 1 else "下跌"

        buy_count = sum(signals.values())
        total_models = len(MODEL_TYPES)
        recommendation = "建議買進" if buy_count >= 2 else "不建議買進"

        if buy_count >= 2:
            summary = f"經過預測後，共有 {buy_count}/{total_models} 個模型推薦買進，明日開高機會高！"
        else:
            summary = f"經過預測後，僅有 {buy_count}/{total_models} 個模型推薦買進，不推薦買進，建議保守觀望。"

        return {
            "success": True,
            "code": code,
            "name": SUPPORTED_STOCKS[code],
            "is_trained": True,
            "buy_count": buy_count,
            "total_models": total_models,
            "recommendation": recommendation,
            "summary": summary,
            "models": model_results,
        }

    except StockPredictionError:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for stock %s", code)
        raise StockPredictionError(
            ErrorType.PREDICTION_FAILED,
            f"{code} 預測失敗：{exc}",
        ) from exc
