import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
import time

# ── Auth & Tracking URI ────────────────────────────────────────────────────────
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

TRACKING_URI = (
    "https://dagshub.com/vrushabh-09/"
    "EndToEnd_MLOps_FastAPI_Water_Potability_Prediction.mlflow"
)
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = "Best Model"

# ── Shared client (created ONCE, reused across all tests) ─────────────────────
_client = MlflowClient()


def load_model_with_retry(model_uri, retries=5, base_delay=10):
    """
    Load an MLflow model with exponential backoff.
    Uses 'models:/' URI when possible to avoid DagsHub run-resolution 500s.
    """
    for attempt in range(retries):
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            if attempt < retries - 1:
                wait = base_delay * (2 ** attempt)   # 10s, 20s, 40s, 80s …
                print(f"[Retry {attempt + 1}/{retries}] Failed – waiting {wait}s. Error: {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Model loading failed after {retries} attempts: {e}") from e


def get_staging_version():
    """Return the latest Staging version object, or None."""
    versions = _client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    return versions[0] if versions else None


class TestModelLoading(unittest.TestCase):

    # ── Fetch staging info ONCE for the whole test class ──────────────────────
    @classmethod
    def setUpClass(cls):
        version = get_staging_version()
        if version is None:
            cls.staging_version = None
            cls.model_uri = None
        else:
            cls.staging_version = version
            # ✅ Use models:/ URI — more stable on DagsHub than runs:/
            cls.model_uri = f"models:/{MODEL_NAME}/Staging"
            # Fallback: cls.model_uri = f"runs:/{version.run_id}/model"

        cls._loaded_model = None   # lazy-loaded once, shared across tests

    def _get_model(self):
        """Load model once and cache it for the test session."""
        if TestModelLoading._loaded_model is None:
            if self.model_uri is None:
                self.fail("No model URI available (no Staging model found).")
            TestModelLoading._loaded_model = load_model_with_retry(self.model_uri)
        return TestModelLoading._loaded_model

    # ── Tests ─────────────────────────────────────────────────────────────────
    def test_model_in_staging(self):
        self.assertIsNotNone(
            self.staging_version,
            f"No model found in Staging for '{MODEL_NAME}'"
        )
        print(f"\nStaging version: {self.staging_version.version} "
              f"| run_id: {self.staging_version.run_id}")

    def test_model_loading(self):
        model = self._get_model()
        self.assertIsNotNone(model)
        print(f"\nModel loaded successfully from '{self.model_uri}'")

    def test_model_performance(self):
        model = self._get_model()

        test_path = "./data/processed/test_processed.csv"
        if not os.path.exists(test_path):
            self.fail(f"Test data not found: {test_path}")

        df = pd.read_csv(test_path)
        X = df.drop(columns=["Potability"])
        y = df["Potability"]

        preds = model.predict(X)

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)

        print("\n── MODEL METRICS ──────────────────────────")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")

        self.assertGreaterEqual(acc,  0.6, "Accuracy below threshold")
        self.assertGreaterEqual(prec, 0.3, "Precision below threshold")
        self.assertGreaterEqual(rec,  0.3, "Recall below threshold")
        self.assertGreaterEqual(f1,   0.3, "F1 below threshold")


if __name__ == "__main__":
    unittest.main()