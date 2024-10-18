import mlflow
import mlflow.sklearn
import subprocess
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Configura MLflow para usar el servidor externo en http://107.23.248.215:5000
mlflow.set_tracking_uri("http://107.23.248.215:5000")

def run_load_data(data_path, output_file):
    """Ejecutar el script load_data_copy.py"""
    command = ["python", "src/load_data_copy.py", data_path]
    subprocess.run(command, check=True)

def run_preprocess_data(data_path, output_train_features, output_test_features, output_train_target, output_test_target):
    """Ejecutar el script preprocess_data.py"""
    command = ["python", "src/preprocess_data.py", data_path, output_train_features, output_test_features, output_train_target, output_test_target]
    subprocess.run(command, check=True)

def run_train_model(X_train_path, y_train_path, model_output_path):
    """Ejecutar el script train.py"""
    command = ["python", "src/train.py", X_train_path, y_train_path, model_output_path]
    subprocess.run(command, check=True)

def run_evaluate_model(model_output_path, X_test_path, y_test_path):
    """Ejecutar el script evaluate.py"""
    model = joblib.load(model_output_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    predictions = model.predict(X_test)
    return predictions, y_test

def run_pipeline(data_path, model_output_path):
    # Iniciar una corrida de MLflow
    with mlflow.start_run(run_name="Oscar's Bone Marrow Experiment"):
        # Establecer un tag con tu nombre
        mlflow.set_tag("Author", "Oscar Becerra A01795611")
        
        print("Cargando datos...")
        run_load_data(data_path, None)

        print("Preprocesando datos...")
        X_train_path = "X_train.csv"
        X_test_path = "X_test.csv"
        y_train_path = "y_train.csv"
        y_test_path = "y_test.csv"
        
        run_preprocess_data(data_path, X_train_path, X_test_path, y_train_path, y_test_path)

        print("Entrenando el modelo...")

        # Registrar parámetros del modelo en MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 2)
        mlflow.log_param("min_samples_leaf", 1)
        mlflow.log_param("random_state", 42)

        run_train_model(X_train_path, y_train_path, model_output_path)

        # Registrar el modelo en el Model Registry y activar el versionado
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_output_path}"
        mlflow.register_model(model_uri, "Bone_Marrow_Model")

        print("Evaluando el modelo...")
        predictions, y_test = run_evaluate_model(model_output_path, X_test_path, y_test_path)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Registrar las métricas en MLflow, pero evitar subir artefactos
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

if __name__ == '__main__':
    # Argumentos: ruta de datos y ruta donde se guardará el modelo
    data_path = "data/raw/bone-marrow.csv"
    model_output_path = "models/random_forest_model.pkl"

    run_pipeline(data_path, model_output_path)
