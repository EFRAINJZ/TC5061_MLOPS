from pathlib import Path
import pandas as pd

def load_data(file_path: str = "data/raw/heart_disease.csv") -> pd.DataFrame:
    """Carga los datos desde local o desde la URL oficial de la UCI."""
    path = Path(file_path)
    if path.exists():
        return pd.read_csv(path)
    
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(data_url, names=column_names, na_values="?")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df