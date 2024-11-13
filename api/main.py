from fastapi import FastAPI, HTTPException
from preprocess_data import preprocess_data
from train import train_model_params
from evaluate import evaluate_model_params
import mlflow
import mlflow.sklearn
import os

app = FastAPI()

# Ruta del archivo CSV en la carpeta api/data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bone-marrow.csv")

@app.post("/run_pipeline")
async def run_pipeline(
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
):
    try:
        # Configuración de MLflowq
        mlflow_host = 'http://ec2-100-29-14-176.compute-1.amazonaws.com:5000/'
        mlflow.set_tracking_uri(mlflow_host)
        experiment = mlflow.set_experiment('BoneMarrow_1')

        # Iniciar un nuevo run en MLflow
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            # Paso 1: Preprocesar los datos
            X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)

            # Loguear parámetros en MLflow
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("random_state", random_state)

            # Paso 2: Entrenar el modelo
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': random_state
            }
            model = train_model_params(X_train, y_train, params)

            # Registrar el modelo en MLflow
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path="model", registered_model_name="BoneMarrow_RF_1"
            )

            # Paso 3: Evaluar el modelo
            accuracy, precision, recall, f1 = evaluate_model_params(model, X_test, y_test)

            # Loguear métricas en MLflow
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1)

        # Retornar las métricas como respuesta JSON
        return {
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "message": "Pipeline executed successfully and results logged in MLflow: http://ec2-100-29-14-176.compute-1.amazonaws.com:5000/"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))