import unittest
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import os
import json
import time
import pandas as pd

# ── Auth ───────────────────────────────────────────────────────────────────────
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# ── Tracking URI (must exactly match model_eval.py & model_reg.py) ────────────
DAGSHUB_URL = "https://dagshub.com"
REPO_OWNER  = "vrushabh-09"
REPO_NAME   = "EndToEnd_MLOps_FastAPI_Water_Potability_Prediction"
TRACKING_URI = f"{DAGSHUB_URL}/{REPO_OWNER}/{REPO_NAME}.mlflow"

mlflow.set_tracking_uri(TRACKING_URI)

# ── Constants (must match model_eval.py & model_reg.py) ───────────────────────
MODEL_NAME      = "Best Model"           # same as run_info.json → model_name
TEST_DATA_PATH  = "./data/processed/test_processed.csv"
RUN_INFO_PATH   = "reports/run_info.json"
TARGET_COLUMN   = "Potability"

# ── Thresholds (same as original test) ────────────────────────────────────────
MIN_ACCURACY   = 0.60
MIN_PRECISION  = 0.30
MIN_RECALL     = 0.30
MIN_F1         = 0.30

# ── Shared MLflow client (ONE instance, reused — avoids hammering DagsHub) ────
_client = MlflowClient()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model_with_retry(model_uri: str, retries: int = 5, base_delay: int = 10):
    """
    Load an MLflow model with exponential backoff.
    DagsHub free tier often returns transient 500s — retrying with
    increasing delays handles them gracefully.
    """
    for attempt in range(retries):
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            if attempt < retries - 1:
                wait = base_delay * (2 ** attempt)   # 10s → 20s → 40s → 80s
                print(
                    f"  [Retry {attempt + 1}/{retries}] load_model failed. "
                    f"Waiting {wait}s…\n  Error: {e}"
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Model loading failed after {retries} attempts: {e}"
                ) from e


def get_staging_version():
    """
    Return the latest registered version in Staging, or None.
    Uses the shared _client to avoid repeated connections.
    """
    versions = _client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    return versions[0] if versions else None


def build_model_uri_from_run_info() -> str:
    """
    Build a runs:/<run_id>/model URI from reports/run_info.json.
    This is the same run_id that model_eval.py writes after its mlflow run,
    and model_reg.py uses to register the model.

    Falls back gracefully to the models:/ URI if the file is missing.
    """
    if os.path.exists(RUN_INFO_PATH):
        with open(RUN_INFO_PATH, "r") as f:
            run_info = json.load(f)
        run_id = run_info.get("run_id")
        if run_id:
            return f"runs:/{run_id}/model"

    # Fallback: stable models:/ URI (resolves Staging version on DagsHub)
    return f"models:/{MODEL_NAME}/Staging"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────────────────────

class TestModelPipeline(unittest.TestCase):
    """
    End-to-end model tests covering:
      1. Staging version exists in the registry
      2. run_info.json is present and well-formed (written by model_eval.py)
      3. Model loads successfully (with retry / exponential backoff)
      4. Model performance meets minimum thresholds
      5. Model artifact is correctly registered and in Staging
    """

    # ── Class-level setup — runs ONCE before any test ─────────────────────────
    @classmethod
    def setUpClass(cls):
        cls.staging_version = get_staging_version()
        cls.model_uri        = build_model_uri_from_run_info()
        cls._loaded_model    = None   # lazy-loaded once, then cached

        print(f"\n{'='*60}")
        print(f"  MODEL  : {MODEL_NAME}")
        print(f"  URI    : {cls.model_uri}")
        if cls.staging_version:
            print(f"  VERSION: {cls.staging_version.version}")
            print(f"  RUN ID : {cls.staging_version.run_id}")
        print(f"{'='*60}\n")

    def _get_model(self):
        """Load model once and cache it — prevents redundant DagsHub calls."""
        if TestModelPipeline._loaded_model is None:
            TestModelPipeline._loaded_model = load_model_with_retry(self.model_uri)
        return TestModelPipeline._loaded_model

    # ── Test 1: Staging version exists ───────────────────────────────────────
    def test_01_model_registered_in_staging(self):
        """model_reg.py must have registered and staged the model."""
        self.assertIsNotNone(
            self.staging_version,
            f"No version of '{MODEL_NAME}' found in Staging. "
            "Run model_reg.py first."
        )
        self.assertIn(
            self.staging_version.current_stage,
            ["Staging"],
            f"Model stage is '{self.staging_version.current_stage}', expected 'Staging'."
        )

    # ── Test 2: run_info.json is valid ───────────────────────────────────────
    def test_02_run_info_json_exists_and_valid(self):
        """model_eval.py writes reports/run_info.json — verify it's well-formed."""
        self.assertTrue(
            os.path.exists(RUN_INFO_PATH),
            f"'{RUN_INFO_PATH}' not found. Run model_eval.py first."
        )
        with open(RUN_INFO_PATH, "r") as f:
            run_info = json.load(f)

        self.assertIn("run_id",     run_info, "run_info.json missing 'run_id' key")
        self.assertIn("model_name", run_info, "run_info.json missing 'model_name' key")
        self.assertEqual(
            run_info["model_name"], MODEL_NAME,
            f"model_name in run_info.json ('{run_info['model_name']}') "
            f"does not match expected '{MODEL_NAME}'"
        )
        self.assertTrue(
            run_info["run_id"],
            "run_id in run_info.json is empty"
        )
        print(f"  run_id    : {run_info['run_id']}")
        print(f"  model_name: {run_info['model_name']}")

    # ── Test 3: Model loads without error ────────────────────────────────────
    def test_03_model_loads_successfully(self):
        """Model artifact must be reachable and loadable from DagsHub."""
        model = self._get_model()
        self.assertIsNotNone(model, "Loaded model is None")
        print(f"\n  Model loaded from: {self.model_uri}")

    # ── Test 4: Model has a predict method ───────────────────────────────────
    def test_04_model_has_predict_method(self):
        """Sanity check: loaded pyfunc model must expose .predict()"""
        model = self._get_model()
        self.assertTrue(
            hasattr(model, "predict"),
            "Loaded model does not have a 'predict' method"
        )

    # ── Test 5: Test data exists and is loadable ──────────────────────────────
    def test_05_test_data_exists(self):
        """test_processed.csv (used by model_eval.py) must be present."""
        self.assertTrue(
            os.path.exists(TEST_DATA_PATH),
            f"Test data not found at '{TEST_DATA_PATH}'"
        )
        df = pd.read_csv(TEST_DATA_PATH)
        self.assertIn(
            TARGET_COLUMN, df.columns,
            f"Target column '{TARGET_COLUMN}' not in test data"
        )
        self.assertGreater(len(df), 0, "Test data is empty")
        print(f"\n  Test data shape: {df.shape}")

    # ── Test 6: Performance thresholds ───────────────────────────────────────
    def test_06_model_performance_meets_thresholds(self):
        """
        Mirrors the metric calculation in model_eval.py exactly.
        Uses zero_division=0 to avoid crashes on edge-case predictions.
        """
        model = self._get_model()

        df = pd.read_csv(TEST_DATA_PATH)
        X  = df.drop(columns=[TARGET_COLUMN])
        y  = df[TARGET_COLUMN]

        preds = model.predict(X)

        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec  = recall_score(y, preds, zero_division=0)
        f1   = f1_score(y, preds, zero_division=0)

        print("\n  ── METRICS ──────────────────────────────")
        print(f"  Accuracy  : {acc:.4f}  (min {MIN_ACCURACY})")
        print(f"  Precision : {prec:.4f}  (min {MIN_PRECISION})")
        print(f"  Recall    : {rec:.4f}  (min {MIN_RECALL})")
        print(f"  F1 Score  : {f1:.4f}  (min {MIN_F1})")
        print("  ─────────────────────────────────────────")

        self.assertGreaterEqual(acc,  MIN_ACCURACY,
            f"Accuracy {acc:.4f} < threshold {MIN_ACCURACY}")
        self.assertGreaterEqual(prec, MIN_PRECISION,
            f"Precision {prec:.4f} < threshold {MIN_PRECISION}")
        self.assertGreaterEqual(rec,  MIN_RECALL,
            f"Recall {rec:.4f} < threshold {MIN_RECALL}")
        self.assertGreaterEqual(f1,   MIN_F1,
            f"F1 Score {f1:.4f} < threshold {MIN_F1}")

    # ── Test 7: Prediction output shape matches input ─────────────────────────
    def test_07_prediction_output_shape(self):
        """Output from model.predict() must have same length as input rows."""
        model = self._get_model()

        df    = pd.read_csv(TEST_DATA_PATH)
        X     = df.drop(columns=[TARGET_COLUMN])
        preds = model.predict(X)

        self.assertEqual(
            len(preds), len(X),
            f"Prediction length {len(preds)} != input rows {len(X)}"
        )

    # ── Test 8: Predictions are valid binary labels ───────────────────────────
    def test_08_predictions_are_valid_binary_labels(self):
        """
        RandomForestClassifier (model_building.py) predicts 0 or 1
        for the Potability column. Verify no garbage outputs.
        """
        model = self._get_model()

        df    = pd.read_csv(TEST_DATA_PATH)
        X     = df.drop(columns=[TARGET_COLUMN])
        preds = model.predict(X)

        unique_preds = set(preds)
        valid_labels = {0, 1}
        self.assertTrue(
            unique_preds.issubset(valid_labels),
            f"Unexpected prediction labels: {unique_preds - valid_labels}"
        )
        print(f"\n  Unique prediction labels: {sorted(unique_preds)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)