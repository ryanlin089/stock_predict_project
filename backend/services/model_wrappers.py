import numpy as np

class KerasWrapper:
    def __init__(self, keras_model, window_size: int, num_features: int):
        self.keras_model = keras_model
        self.window_size = window_size
        self.num_features = num_features

    def predict(self, X):
        X_3d = X.reshape(-1, self.window_size, self.num_features)
        probs = self.keras_model.predict(X_3d, verbose=0)
        return (probs > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        X_3d = X.reshape(-1, self.window_size, self.num_features)
        probs = self.keras_model.predict(X_3d, verbose=0)
        return np.hstack([1.0 - probs, probs])
