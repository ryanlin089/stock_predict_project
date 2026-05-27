from pathlib import Path
import os

# 降低 TensorFlow 一般 warning，但不吞掉 Python exception。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import keras
from keras import layers

from services.constants import CHECKPOINT_DIR
from services.model_wrappers import KerasWrapper


def _load_training_data(file_path: str | Path, selected_features: list[str], window_size: int, test_split: float):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    scaler = StandardScaler()
    raw_split_index = int(len(df) * (1 - test_split))
    scaler.fit(df.iloc[:raw_split_index][selected_features])
    scaled_features = scaler.transform(df[selected_features])

    X = []
    y = []
    labels = df["Label"].values

    for i in range(len(scaled_features) - window_size + 1):
        window_data = scaled_features[i : i + window_size]
        X.append(window_data.flatten())
        y.append(labels[i + window_size - 1])

    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("訓練集或測試集為空，請確認資料量是否足夠")

    return X_train, X_test, y_train, y_test, scaler


def prepare_data_and_train_xgb(
    file_path: str | Path,
    selected_features: list[str],
    window_size: int = 3,
    test_split: float = 0.2,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    X_train, X_test, y_train, y_test, scaler = _load_training_data(
        file_path, selected_features, window_size, test_split
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=300,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost 測試集準確率: {acc:.4f}")

    stock_code = Path(file_path).stem.split("_")[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint_dir / f"{stock_code}_xgb.joblib")
    joblib.dump(scaler, checkpoint_dir / f"{stock_code}_xgb_scaler.joblib")

    return model, scaler


def prepare_data_and_train_rnn(
    file_path: str | Path,
    selected_features: list[str],
    window_size: int = 3,
    test_split: float = 0.2,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    X_train, X_test, y_train, y_test, scaler = _load_training_data(
        file_path, selected_features, window_size, test_split
    )
    num_features = len(selected_features)
    X_train_3d = X_train.reshape(-1, window_size, num_features)

    keras_model = keras.Sequential(
        [
            layers.Input(shape=(window_size, num_features)),
            layers.SimpleRNN(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    keras_model.fit(X_train_3d, y_train, epochs=30, batch_size=16, verbose=0)

    wrapped_model = KerasWrapper(keras_model, window_size, num_features)
    y_pred = wrapped_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RNN 測試集準確率: {acc:.4f}")

    stock_code = Path(file_path).stem.split("_")[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    keras_model.save(checkpoint_dir / f"{stock_code}_rnn.keras")
    joblib.dump(scaler, checkpoint_dir / f"{stock_code}_rnn_scaler.joblib")

    return wrapped_model, scaler


def prepare_data_and_train_lstm(
    file_path: str | Path,
    selected_features: list[str],
    window_size: int = 3,
    test_split: float = 0.2,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    X_train, X_test, y_train, y_test, scaler = _load_training_data(
        file_path, selected_features, window_size, test_split
    )
    num_features = len(selected_features)
    X_train_3d = X_train.reshape(-1, window_size, num_features)

    keras_model = keras.Sequential(
        [
            layers.Input(shape=(window_size, num_features)),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    keras_model.fit(X_train_3d, y_train, epochs=30, batch_size=16, verbose=0)

    wrapped_model = KerasWrapper(keras_model, window_size, num_features)
    y_pred = wrapped_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"LSTM 測試集準確率: {acc:.4f}")

    stock_code = Path(file_path).stem.split("_")[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    keras_model.save(checkpoint_dir / f"{stock_code}_lstm.keras")
    joblib.dump(scaler, checkpoint_dir / f"{stock_code}_lstm_scaler.joblib")

    return wrapped_model, scaler


def build_transformer(window_size: int, num_features: int):
    inputs = layers.Input(shape=(window_size, num_features))
    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ff_output = layers.Dense(32, activation="relu")(out1)
    ff_output = layers.Dense(num_features)(ff_output)
    ff_output = layers.Dropout(0.1)(ff_output)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff_output)

    flat = layers.Flatten()(out2)
    dense1 = layers.Dense(16, activation="relu")(flat)
    dense1 = layers.Dropout(0.2)(dense1)
    outputs = layers.Dense(1, activation="sigmoid")(dense1)
    return keras.Model(inputs=inputs, outputs=outputs)


def prepare_data_and_train_transformer(
    file_path: str | Path,
    selected_features: list[str],
    window_size: int = 3,
    test_split: float = 0.2,
    checkpoint_dir: Path = CHECKPOINT_DIR,
):
    X_train, X_test, y_train, y_test, scaler = _load_training_data(
        file_path, selected_features, window_size, test_split
    )
    num_features = len(selected_features)
    X_train_3d = X_train.reshape(-1, window_size, num_features)

    keras_model = build_transformer(window_size, num_features)
    keras_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    keras_model.fit(X_train_3d, y_train, epochs=30, batch_size=16, verbose=0)

    wrapped_model = KerasWrapper(keras_model, window_size, num_features)
    y_pred = wrapped_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Transformer 測試集準確率: {acc:.4f}")

    stock_code = Path(file_path).stem.split("_")[0]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    keras_model.save(checkpoint_dir / f"{stock_code}_transformer.keras")
    joblib.dump(scaler, checkpoint_dir / f"{stock_code}_transformer_scaler.joblib")

    return wrapped_model, scaler
