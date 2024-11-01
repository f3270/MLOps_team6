import pytest
from src.preprocess_data import preprocess_data
from src.train import train_model_params
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "../data/raw/bone-marrow.csv"

@pytest.fixture
def sample_data():
    X_train, _, y_train, _ = preprocess_data(DATA_PATH)
    return X_train, y_train

@pytest.fixture
def model_params():
    # Configuración de hiperparámetros para el modelo
    return {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }

def test_train_model_output_type(sample_data, model_params):
    X_train, y_train = sample_data
    model = train_model_params(X_train, y_train, model_params)
    assert isinstance(model, RandomForestClassifier), "El modelo entrenado no es un RandomForestClassifier"

def test_train_model_accuracy(sample_data, model_params):
    X_train, y_train = sample_data
    model = train_model_params(X_train, y_train, model_params)
    accuracy = model.score(X_train, y_train)
    assert accuracy >= 0.8, f"El modelo tiene una precisión menor a la esperada: {accuracy:.2f}"
