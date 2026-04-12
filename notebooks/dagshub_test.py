import dagshub
import mlflow

mlflow.set_tracking_uri="https://dagshub.com/vrushabh-09/EndToEnd_MLOps_FastAPI_Water_Potability_Prediction.mlflow"
dagshub.init(repo_owner='vrushabh-09', repo_name='EndToEnd_MLOps_FastAPI_Water_Potability_Prediction', mlflow=True)


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)