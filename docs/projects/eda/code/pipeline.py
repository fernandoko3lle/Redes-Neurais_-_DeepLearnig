"""Pipeline de pré-processamento do Stroke Prediction Dataset.

Módulo importável: define o `ColumnTransformer` usado pelo EDA (item 4) e,
mais tarde, pela entrega de Classificação — sempre ajustado só no treino.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "stroke"
DROP_COLUMNS = ["id"]  # identificador, não carrega informação preditiva

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]


def build_preprocessor() -> ColumnTransformer:
    """Mediana + padronização para numéricas; moda + one-hot para categóricas."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
