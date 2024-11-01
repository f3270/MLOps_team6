import pytest
from src.evaluate import evaluate_model_params
from src.preprocess_data import preprocess_data
from src.train import train_model_params

DATA_PATH = "../data/raw/bone-marrow.csv"

model_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

@pytest.fixture
def trained_model():
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
    model = train_model_params(X_train, y_train, model_params)
    return model, X_test, y_test

def test_evaluate_model_metrics(trained_model):
    model, X_test, y_test = trained_model
    accuracy, precision, recall, f1 = evaluate_model_params(model, X_test, y_test)

    # Verificar que cada métrica esté en un rango permitido
    assert 0 <= accuracy <= 1, f"Accuracy fuera de rango: {accuracy}"
    assert 0 <= precision <= 1, f"Precision fuera de rango: {precision}"
    assert 0 <= recall <= 1, f"Recall fuera de rango: {recall}"
    assert 0 <= f1 <= 1, f"F1 Score fuera de rango: {f1}"

def test_evaluate_model_output_type(trained_model):
    model, X_test, y_test = trained_model
    accuracy, precision, recall, f1 = evaluate_model_params(model, X_test, y_test)

    # Verificar que cada métrica sea un número flotante
    assert isinstance(accuracy, float), "Accuracy no es de tipo float"
    assert isinstance(precision, float), "Precision no es de tipo float"
    assert isinstance(recall, float), "Recall no es de tipo float"
    assert isinstance(f1, float), "F1 Score no es de tipo float"