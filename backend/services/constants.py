from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

SUPPORTED_STOCKS = {
    "2303": "聯電",
    "2344": "華邦電",
    "3481": "群創光電",
    "6770": "力積電",
    "8028": "昇陽國際半導體",
}

MODEL_TYPES = ["xgb", "rnn", "lstm", "transformer"]
DAYS_TO_LOOK_BACK = 7
TRAIN_TEST_SPLIT = 0.2

# 必須與 preprocessing 輸出的欄位順序一致。
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Price_Change_Pct",
    "Close_Relative_Pos",
    "Volume_Ratio",
    "Volume_Ratio_5D",
    "True_Pct_Change",
    "MA5",
    "Bias_MA5",
    "MA20",
    "Bias_MA20",
]
