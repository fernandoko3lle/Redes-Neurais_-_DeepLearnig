"""Exercise 3 — Preparando dados reais (Spaceship Titanic) para uma rede tanh.

Requer o arquivo ``train.csv`` da competição Kaggle "Spaceship Titanic"
(https://www.kaggle.com/competitions/spaceship-titanic/data), colocado em
``docs/exercises/data/code/data/train.csv`` (baixe manualmente — é preciso
login no Kaggle; não há como baixar por script sem credenciais).

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise3_preprocessing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[0] / "figures"
DATA_PATH = HERE / "data" / "train.csv"

NUMERIC_COLS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP_COLS = ["Cabin", "Name", "PassengerId"]
TARGET = "Transported"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {DATA_PATH}.\n"
            "Baixe train.csv em https://www.kaggle.com/competitions/spaceship-titanic/data "
            "e salve nesse caminho antes de rodar este script."
        )
    return pd.read_csv(DATA_PATH)


def describe_data(df: pd.DataFrame) -> None:
    positive_rate = df[TARGET].mean()
    print(f"Proporção da classe positiva em '{TARGET}': {positive_rate:.4f} ({positive_rate * 100:.2f}%)")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"faltantes": missing, "percentual": missing_pct})
    missing_table = missing_table[missing_table["faltantes"] > 0].sort_values("faltantes", ascending=False)
    print("\nValores faltantes por coluna:")
    print(missing_table)

    print("\nMédia, mediana e máximo das colunas de gasto (antes de transformar):")
    print(df[SPEND_COLS].agg(["mean", "median", "max"]).T)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    train_df = train_df.drop(columns=DROP_COLS).copy()
    test_df = test_df.drop(columns=DROP_COLS).copy()

    # --- imputação: mediana para numéricas, moda para categóricas ----------
    num_imputer = SimpleImputer(strategy="median")
    train_df[NUMERIC_COLS] = num_imputer.fit_transform(train_df[NUMERIC_COLS])
    test_df[NUMERIC_COLS] = num_imputer.transform(test_df[NUMERIC_COLS])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_df[CATEGORICAL_COLS] = cat_imputer.fit_transform(train_df[CATEGORICAL_COLS])
    test_df[CATEGORICAL_COLS] = cat_imputer.transform(test_df[CATEGORICAL_COLS])

    # --- feature de engenharia: gasto total ---------------------------------
    train_df["TotalSpend"] = train_df[SPEND_COLS].sum(axis=1)
    test_df["TotalSpend"] = test_df[SPEND_COLS].sum(axis=1)

    # --- cauda pesada: log1p nas colunas de gasto (+ TotalSpend) ------------
    heavy_tail_cols = SPEND_COLS + ["TotalSpend"]
    foodcourt_before = train_df["FoodCourt"].copy()
    for col in heavy_tail_cols:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])
    foodcourt_after = train_df["FoodCourt"].copy()

    # --- one-hot nas categóricas; handle_unknown ignora categoria nova ------
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = encoder.fit_transform(train_df[CATEGORICAL_COLS])
    test_cat = encoder.transform(test_df[CATEGORICAL_COLS])
    cat_names = encoder.get_feature_names_out(CATEGORICAL_COLS)

    # --- escalonamento para [-1, 1], compatível com a saída de tanh --------
    scale_cols = NUMERIC_COLS + ["TotalSpend"]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_num = scaler.fit_transform(train_df[scale_cols])
    test_num = scaler.transform(test_df[scale_cols])

    X_train = np.hstack([train_num, train_cat])
    X_test = np.hstack([test_num, test_cat])
    feature_names = scale_cols + list(cat_names)

    plot_figure6(foodcourt_before, foodcourt_after)

    print(f"\nShape final da matriz de features de treino: {X_train.shape}")
    print(f"Shape final da matriz de features de teste:   {X_test.shape}")
    print(f"NaN remanescentes no treino: {np.isnan(X_train).sum()}")
    print(f"NaN remanescentes no teste:  {np.isnan(X_test).sum()}")
    print(f"Treino  — min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"Teste   — min={X_test.min():.4f}, max={X_test.max():.4f}")
    print(f"Features: {feature_names}")

    return X_train, X_test, train_df, test_df


def plot_figure6(before: pd.Series, after: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(before, bins=40, color="tab:blue")
    axes[0].set_title("FoodCourt — antes (bruto)")
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("Contagem")

    axes[1].hist(after, bins=40, color="tab:orange")
    axes[1].set_title("FoodCourt — depois (log1p)")
    axes[1].set_xlabel("log(1 + FoodCourt)")
    axes[1].set_ylabel("Contagem")

    fig.suptitle("Figura 6 — Efeito do log1p em uma feature de cauda pesada")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-foodcourt-log.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    describe_data(df)
    train_df, test_df = split_data(df)
    print(f"\nAmostras de treino: {len(train_df)} | Amostras de teste: {len(test_df)}")
    preprocess(train_df, test_df)


if __name__ == "__main__":
    main()
