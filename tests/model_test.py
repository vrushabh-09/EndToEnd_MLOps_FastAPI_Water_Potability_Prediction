import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
import time

# Load DagsHub token
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# MLflow auth
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/vrushabh-09/EndToEnd_MLOps_FastAPI_Water_Potability_Prediction.mlflow"
)

model_name = "Best Model"


class TestModelLoading(unittest.TestCase):

    def load_model_with_retry(self, model_uri, retries=3, delay=5):
        """
        Retry logic for loading model (handles DagsHub 500 errors)
        """
        for attempt in range(retries):
            try:
                return mlflow.pyfunc.load_model(model_uri)
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Retry {attempt+1} failed. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Model loading failed after retries: {e}")

    def test_model_in_staging(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        self.assertGreater(len(versions), 0, "No model found in Staging")

    def test_model_loading(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.fail("No model in Staging")

        run_id = versions[0].run_id
        model_uri = f"runs:/{run_id}/model"

        # retry logic
        loaded_model = self.load_model_with_retry(model_uri)

        self.assertIsNotNone(loaded_model)
        print(f"Model loaded successfully from {model_uri}")

    def test_model_performance(self):
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.fail("No model in Staging")

        run_id = versions[0].run_id
        model_uri = f"runs:/{run_id}/model"

        # retry logic
        model = self.load_model_with_retry(model_uri)

        # Load test data
        test_path = "./data/processed/test_processed.csv"
        if not os.path.exists(test_path):
            self.fail(f"Test data not found: {test_path}")

        df = pd.read_csv(test_path)
        X = df.drop(columns=["Potability"])
        y = df["Potability"]

        preds = model.predict(X)

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        f1 = f1_score(y, preds)

        print("\nMODEL METRICS")
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {f1}")

        # Threshold checks
        self.assertGreaterEqual(acc, 0.6)
        self.assertGreaterEqual(prec, 0.3)
        self.assertGreaterEqual(rec, 0.3)
        self.assertGreaterEqual(f1, 0.3)


if __name__ == "__main__":
    unittest.main()