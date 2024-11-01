import pytest
from src.preprocess_data import preprocess_data
import pandas as pd

# Ejemplo de datos de prueba en formato CSV
DATA_PATH = "../data/raw/bone-marrow.csv"

def test_preprocess_data_output_shape():
    """Verifica que la función devuelva el número correcto de conjuntos y tamaños."""
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)

    # Verificar que los datos de salida no estén vacíos
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    # Verifica la proporción de la división de datos
    total_data_points = len(X_train) + len(X_test)
    assert len(X_train) / total_data_points == pytest.approx(0.8, rel=0.05)
    assert len(X_test) / total_data_points == pytest.approx(0.2, rel=0.05)


def test_handle_missing_values():
    """Verifica que la función maneje valores nulos correctamente."""
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)

    # Verifica si hay valores nulos en X_train
    missing_columns = X_train.columns[X_train.isnull().any()]
    if len(missing_columns) > 0:
        print("\nColumnas con valores faltantes en X_train:", missing_columns)
        print("\nNúmero de valores nulos por columna en X_train:\n", X_train[missing_columns].isnull().sum())
        print("\nFilas con valores faltantes en X_train:\n", X_train[X_train.isnull().any(axis=1)])

    # Test que falla si hay valores nulos
    assert not X_train.isnull().values.any()

def test_column_names():
    """Confirma que las columnas de salida son las esperadas."""
    X_train, X_test, _, _ = preprocess_data(DATA_PATH)

    # Lista de columnas esperadas después del procesamiento
    expected_columns = ['Recipientgender', 'Stemcellsource', 'Donorage', 'Donorage35', 'IIIV', 'Gendermatch', 'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 'CMVstatus', 'DonorCMV', 'RecipientCMV', 'Riskgroup', 'Txpostrelapse', 'HLAmatch', 'HLAmismatch', 'Antigen', 'Allele', 'HLAgrI', 'Recipientage', 'Recipientage10', 'Recipientageint', 'Relapse', 'aGvHDIIIIV', 'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass', 'ANCrecovery', 'PLTrecovery', 'survival_time', 'Disease_ALL', 'Disease_AML', 'Disease_chronic', 'Disease_lymphoma', 'Disease_nonmalignant']
    assert list(X_train.columns) == expected_columns
    assert list(X_test.columns) == expected_columns