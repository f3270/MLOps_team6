import pandas as pd
import sys

def load_data(filepath):
    """Carga los datos desde un archivo CSV"""
    return pd.read_csv(filepath)

if __name__ == '__main__':
    data_path = sys.argv[1]
    data = load_data(data_path)