from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from services.constants import DATA_DIR, FEATURE_COLUMNS
from services.errors import ErrorType, StockPredictionError


def data_preprocessing(stock: str, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """
    讀取 data/{stock}__D.txt，產生：
    - data/{stock}_train.csv
    - data/{stock}_test.csv

    回傳：
    - train_csv_path
    - test_csv_path
    """
    year_start = 2025
    year_end = 2026
    train_data_scale = 0.8

    data_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = data_dir / f"{stock}__D.txt"
    train_output_path = data_dir / f"{stock}_train.csv"
    test_output_path = data_dir / f"{stock}_test.csv"

    if not raw_file_path.exists():
        raise StockPredictionError(
            ErrorType.DATA_FILE_NOT_FOUND,
            f"找不到原始股價資料：{raw_file_path}",
        )

    # 原始資料是 tab 分隔文字檔；前兩行是商品資訊與中文欄位列前置資訊。
    df = pd.read_csv(raw_file_path, sep="\t", skiprows=2)

    required_cols = ["日期", "開盤價", "最高價", "最低價", "收盤價", "成交量"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise StockPredictionError(
            ErrorType.DATA_FILE_NOT_FOUND,
            f"原始資料欄位不完整，缺少：{', '.join(missing_cols)}",
        )

    df["日期"] = df["日期"].astype(str).str.replace("'", "", regex=False)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).reset_index(drop=True)

    df = df.rename(
        columns={
            "日期": "Date",
            "開盤價": "Open",
            "最高價": "High",
            "最低價": "Low",
            "收盤價": "Close",
            "成交量": "Volume",
        }
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    df["Prev_Close"] = df["Close"].shift(1)
    df["Price_Change_Pct"] = (df["High"] - df["Low"]) / df["Low"]

    df["Close_Relative_Pos"] = np.where(
        df["High"] == df["Low"],
        np.where(df["Close"] >= df["Prev_Close"], 1.0, 0.0),
        (df["Close"] - df["Low"]) / (df["High"] - df["Low"]),
    )

    df["Volume_Ratio"] = df["Volume"] / df["Volume"].shift(1)
    df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()
    df["Volume_Ratio_5D"] = df["Volume"] / df["Volume_MA5"]

    df["True_Pct_Change"] = (df["Close"] - df["Prev_Close"]) / df["Prev_Close"]
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["Bias_MA5"] = (df["Close"] - df["MA5"]) / df["MA5"]
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Bias_MA20"] = (df["Close"] - df["MA20"]) / df["MA20"]

    df["Next_Open"] = df["Open"].shift(-1)
    df["Label"] = np.where(
        df["Next_Open"].isna(),
        -1,
        np.where(df["Next_Open"] > df["Close"] * 1.005, 1, 0),
    )

    df = df[(df["Date"].dt.year >= year_start) & (df["Date"].dt.year <= year_end)]
    df = df.drop(columns=["Prev_Close", "Volume_MA5", "Next_Open"])

    # 8028 等資料可能出現成交量為 0，會造成 inf；這裡明確清理，不讓 scaler 訓練失敗。
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    output_columns = ["Date", *FEATURE_COLUMNS, "Label"]
    df = df[output_columns]

    if len(df) < 30:
        raise StockPredictionError(
            ErrorType.DATA_FILE_NOT_FOUND,
            f"{stock} 在 {year_start}~{year_end} 的可用資料不足，無法切分訓練與測試資料",
        )

    split_idx = int(len(df) * train_data_scale)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    return train_output_path, test_output_path


def pearson_correlation(file_path: str | Path, save_heatmap: bool = False) -> list[int]:
    """
    輸入：train csv path
    輸出：與 FEATURE_COLUMNS 同順序的一組 one-hot vector。
    """
    df = pd.read_csv(file_path)
    numeric_df = df.drop(columns=["Date"])
    corr_matrix = numeric_df.corr(method="pearson")

    if save_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Pearson Correlation Heatmap")
        plt.tight_layout()
        output_path = Path(file_path).with_name(f"{Path(file_path).stem}_correlation_heatmap.png")
        plt.savefig(output_path)
        plt.close()

    label_corr = corr_matrix["Label"].drop("Label")
    threshold = 0.05
    return [1 if abs(label_corr.get(feature, 0)) >= threshold else 0 for feature in FEATURE_COLUMNS]


def get_selected_features(train_csv_path: str | Path) -> list[str]:
    one_hot_vector = pearson_correlation(train_csv_path)
    selected_features = [feature for feature, keep in zip(FEATURE_COLUMNS, one_hot_vector) if keep == 1]

    if not selected_features:
        raise ValueError("Pearson correlation 沒有選出任何特徵，請調整 THRESHOLD 或檢查資料")

    return selected_features
