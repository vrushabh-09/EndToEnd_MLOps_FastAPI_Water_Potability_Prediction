import mlflow
from mlflow.tracking import MlflowClient
import os

# ── Auth ───────────────────────────────────────────────────────────────────────
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# ── Tracking URI (matches model_eval.py & model_reg.py) ───────────────────────
dagshub_url = "https://dagshub.com"
repo_owner  = "vrushabh-09"
repo_name   = "EndToEnd_MLOps_FastAPI_Water_Potability_Prediction"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "Best Model"   # must match model_eval.py & model_reg.py


def promote_model_to_production():
    """
    Promote the latest Staging model to Production.
    Archives any existing Production version first.
    """
    client = MlflowClient()

    # ── Step 1: Get latest Staging version ────────────────────────────────────
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print(f"[WARN] No model found in 'Staging' for '{model_name}'. Aborting.")
        return

    latest_staging         = staging_versions[0]
    staging_version_number = latest_staging.version
    staging_run_id         = latest_staging.run_id
    print(f"[INFO] Found Staging model  → version: {staging_version_number} | run_id: {staging_run_id}")

    # ── Step 2: Archive any existing Production version ───────────────────────
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    if production_versions:
        current_prod         = production_versions[0]
        prod_version_number  = current_prod.version
        client.transition_model_version_stage(
            name=model_name,
            version=prod_version_number,
            stage="Archived",
            archive_existing_versions=False,
        )
        print(f"[INFO] Archived Production model → version: {prod_version_number}")
    else:
        print("[INFO] No model currently in 'Production' — skipping archive step.")

    # ── Step 3: Promote Staging → Production ──────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version_number,
        stage="Production",
        archive_existing_versions=False,
    )
    print(f"[INFO] Promoted model version {staging_version_number} → 'Production'. ✅")


if __name__ == "__main__":
    promote_model_to_production() 