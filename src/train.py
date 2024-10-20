import pandas as pd
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model_params(X_train, y_train, params):
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators'),
        max_depth=params.get('max_depth'),
        min_samples_split=params.get('min_samples_split'),
        min_samples_leaf=params.get('min_samples_leaf'),
        random_state=params.get('random_state')
    )
    model.fit(X_train, y_train)
    return model

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())
    return model


if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(X_train_path, y_train_path)
    joblib.dump(model, model_path)