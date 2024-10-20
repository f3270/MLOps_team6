from preprocess_data import preprocess_data
from train import train_model_params
from evaluate import evaluate_model_params

import mlflow

if __name__ == '__main__':
    mlflow_host = 'http://ec2-35-153-203-1.compute-1.amazonaws.com:5000/'

    mlflow.set_tracking_uri(mlflow_host)
    experiment = mlflow.set_experiment('BoneMarrow_1')

    filepath=r'../data/raw/bone-marrow.csv'

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        X_train, X_test, y_train, y_test = preprocess_data(filepath)

        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        mlflow.log_param("n_estimators", params.get('n_estimators'))
        mlflow.log_param("max_depth", params.get('max_depth'))
        mlflow.log_param("min_samples_split", params.get('min_samples_split'))
        mlflow.log_param("min_samples_leaf", params.get('min_samples_leaf'))
        mlflow.log_param("random_state", params.get('random_state'))

        model = train_model_params(X_train, y_train, params)
        # model_info = mlflow.sklearn.log_model(
        #     sk_model=model, artifact_path="model", registered_model_name="BoneMarrow_RF_1"
        # )
        # mlflow.register_model(model, "BoneMarrow_RF_1")

        accuracy, precision, recall, f1 = evaluate_model_params(model, X_test, y_test)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)
