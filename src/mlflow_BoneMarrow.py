from preprocess_data import preprocess_data
from train import train_model_params
from evaluate import evaluate_model_params

# Main
def main(filepath):
    X_train, X_test, y_train, y_test = preprocess_data(filepath)

    params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }

    model = train_model_params(X_train, y_train, params)
    accuracy, precision, recall, f1 = evaluate_model_params(model, X_test, y_test)


if __name__ == '__main__':
    main(filepath=r'../data/raw/bone-marrow.csv')