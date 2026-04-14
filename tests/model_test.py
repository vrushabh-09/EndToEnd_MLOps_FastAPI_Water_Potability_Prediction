import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd


# Load DagsHub token
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/vrushabh-09/EndToEnd_MLOps_FastAPI_Water_Potability_Prediction.mlflow"
)

# Model name
model_name = "Best Model"

# Initialize MLflow client
client = MlflowClient()


class TestModelLoading(unittest.TestCase):
    """
    Unit test class to verify MLflow model loading from Staging stage
    """

    def test_model_in_staging(self):
        """
        Test if model exists in Staging stage
        """
        self.versions = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )

        self.assertGreater(
            len(self.versions),
            0,
            "No model found in the 'Staging' stage."
        )

    def test_model_loading(self):
        """
        Test if model loads successfully from Staging
        """

        versions = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )

        if len(versions) == 0:
            self.fail("No model found in the Staging stage, skipping model loading test.")

        # Get latest version details
        latest_version = versions[0]
        run_id = latest_version.run_id

        # IMPORTANT: correct artifact path
        logged_model = f"runs:/{run_id}/model"

        try:
            # Try loading model
            loaded_model = mlflow.pyfunc.load_model(logged_model)

        except Exception as e:
            self.fail(f"Failed to load the model: {e}")

        # Validate model
        self.assertIsNotNone(
            loaded_model,
            "The loaded model is None."
        )

        print(f"Model successfully loaded from {logged_model}.")


# Run tests
if __name__ == "__main__":
    unittest.main()