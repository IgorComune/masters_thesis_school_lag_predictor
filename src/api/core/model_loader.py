import joblib
from pathlib import Path

class ModelService:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        self.model = joblib.load(self.model_path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
