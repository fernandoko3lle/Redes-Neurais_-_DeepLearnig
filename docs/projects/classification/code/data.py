"""Dados da entrega de Classificação — reaproveita o pipeline do EDA sem alterações.

Mesmo dataset, mesma tarefa e mesmo split treino/teste do EDA (semente 42). A
validação sai de dentro do treino do EDA (mais um split 80/20 estratificado,
mesma semente).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EDA_CODE = HERE.parents[2] / "projects" / "eda" / "code"
sys.path.insert(0, str(EDA_CODE))
from pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, build_preprocessor  # noqa: E402
from eda import cap_outliers  # noqa: E402  (mesma correção de outliers apontada no EDA)

DATA_PATH = HERE / "data" / "healthcare-dataset-stroke-data.csv"


def load_splits():
    """Devolve (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, preprocessor)."""
    df = pd.read_csv(DATA_PATH)

    # Mesmo split treino/teste do EDA (80/20 estratificado, semente 42).
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
    # Validação sai de dentro do treino (80/20 estratificado, mesma semente).
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df[TARGET], random_state=42)

    # Correção do EDA: cap de outliers (IQR*1.5) em avg_glucose_level/bmi, ajustado só no treino.
    train_df, val_df, _ = cap_outliers(train_df, val_df, ["avg_glucose_level", "bmi"])
    train_df, test_df, _ = cap_outliers(train_df, test_df, ["avg_glucose_level", "bmi"])

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df).astype(np.float32)
    X_val = preprocessor.transform(val_df).astype(np.float32)
    X_test = preprocessor.transform(test_df).astype(np.float32)
    feature_names = list(preprocessor.get_feature_names_out())

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }
