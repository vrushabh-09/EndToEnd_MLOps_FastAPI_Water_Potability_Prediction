import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os


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

# Model name (IMPORTANT: must match registry)
model_name = "Best Model"

client = MlflowClient()


class TestModelLoading(unittest.TestCase):
    """
    Unit test class to verify MLflow model loading from Staging stage
    """

    def test_model_in_staging(self):
        """
        Check if model exists in Staging
        """
        versions = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )

        self.assertGreater(
            len(versions),
            0,
            "No model found in the 'Staging' stage."
        )

    def test_model_loading(self):
        """
        Load model from MLflow Registry (Staging)
        """

        model_uri = f"models:/{model_name}/Staging"

        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)

        except Exception as e:
            self.fail(f"Failed to load the model: {e}")

        # Validate model
        self.assertIsNotNone(
            loaded_model,
            "The loaded model is None."
        )

        print(f"Model successfully loaded from {model_uri}.")


if __name__ == "__main__":
    unittest.main()