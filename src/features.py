import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_preprocessor() -> ColumnTransformer:
    """Construye un preprocesador para variables numéricas y categóricas."""
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_features = ["ca", "thal"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    passthrough_features = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("pass", "passthrough", passthrough_features),
    ])
    return preprocessor

def extract_target_and_features(df: pd.DataFrame):
    """Limpia duplicados, binariza el target y separa X e y."""
    df_clean = df.drop_duplicates().copy()
    df_clean["target"] = df_clean["target"].apply(lambda x: 1 if x > 0 else 0)
    y = df_clean["target"]
    X = df_clean.drop(columns=["target"])
    return X, y