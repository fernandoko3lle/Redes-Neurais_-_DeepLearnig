"""EDA — Stroke Prediction Dataset.

Roda as etapas 1 a 4 do enunciado: inspeção inicial, análise univariada,
análise bivariada/multivariada e pré-processamento (pipeline + PCA/t-SNE/UMAP).
Salva as figuras em ``figures/`` e imprime todos os números do relatório.

Uso (a partir da raiz do repositório):

    python docs/projects/eda/code/eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

try:
    from umap import UMAP
except ImportError:
    UMAP = None  # ambiente sem o numba funcionando; rode este script numa máquina
    # com umap-learn disponível para gerar a parte de UMAP da Figura 7/8.

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, build_preprocessor  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
DATA_PATH = HERE / "data" / "healthcare-dataset-stroke-data.csv"
RNG = np.random.default_rng(42)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {DATA_PATH}.\n"
            "Baixe o dataset em https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset "
            "e salve o CSV nesse caminho."
        )
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------------------------------- #
# Etapa 1 — Inspeção inicial
# --------------------------------------------------------------------------- #

def step1_inspection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Etapa 1 — Inspeção inicial ===")
    print(f"Shape bruto: {df.shape}")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"faltantes": missing, "percentual": missing_pct})
    missing_table = missing_table[missing_table["faltantes"] > 0]
    print("\nValores faltantes:")
    print(missing_table)

    duplicates = df.duplicated().sum()
    id_duplicates = df["id"].duplicated().sum()
    print(f"\nLinhas duplicadas (todas as colunas): {duplicates}")
    print(f"IDs duplicados: {id_duplicates}")

    for col in CATEGORICAL_FEATURES:
        print(f"Categorias de {col}: {sorted(df[col].astype(str).unique())}")

    balance = df[TARGET].value_counts(normalize=True) * 100
    print(f"\nBalanceamento do alvo '{TARGET}' (%):\n{balance}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )
    print(f"\nTreino: {len(train_df)} amostras | Teste: {len(test_df)} amostras")

    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color=["tab:blue", "tab:red"])
    ax.set_xticklabels(["Sem AVC (0)", "Com AVC (1)"], rotation=0)
    ax.set_ylabel("Contagem")
    ax.set_title("Figura 1 — Balanceamento do alvo (stroke)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-target-balance.png", dpi=150)
    plt.close(fig)

    return train_df, test_df


# --------------------------------------------------------------------------- #
# Etapa 2 — Análise univariada
# --------------------------------------------------------------------------- #

def step2_univariate(train_df: pd.DataFrame) -> None:
    print("\n=== Etapa 2 — Análise univariada ===")

    desc = train_df[NUMERIC_FEATURES].describe().T
    print("\nEstatísticas descritivas (numéricas, treino):")
    print(desc)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        ax.hist(train_df[col].dropna(), bins=30, color="tab:blue")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Contagem")
    fig.suptitle("Figura 2 — Distribuição das features numéricas (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-numeric-distributions.png", dpi=150)
    plt.close(fig)

    print("\nFrequência e cardinalidade (categóricas, treino):")
    for col in CATEGORICAL_FEATURES:
        counts = train_df[col].value_counts()
        print(f"\n{col} (cardinalidade={train_df[col].nunique()}):")
        print(counts)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        train_df[col].value_counts().plot(kind="bar", ax=ax, color="tab:orange")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Figura 3 — Frequência das features categóricas (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-categorical-frequencies.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Etapa 3 — Análise bivariada e multivariada
# --------------------------------------------------------------------------- #

def step3_bivariate(train_df: pd.DataFrame) -> None:
    print("\n=== Etapa 3 — Análise bivariada e multivariada ===")

    corr = train_df[NUMERIC_FEATURES].corr(method="spearman")
    print("\nCorrelação de Spearman entre numéricas (treino):")
    print(corr)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(NUMERIC_FEATURES)))
    ax.set_yticks(range(len(NUMERIC_FEATURES)))
    ax.set_xticklabels(NUMERIC_FEATURES, rotation=45)
    ax.set_yticklabels(NUMERIC_FEATURES)
    for i in range(len(NUMERIC_FEATURES)):
        for j in range(len(NUMERIC_FEATURES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, label="Spearman")
    ax.set_title("Figura 4 — Correlação entre numéricas (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-numeric-correlation.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["hypertension", "heart_disease", "smoking_status"]):
        rate = train_df.groupby(col)[TARGET].mean() * 100
        rate.plot(kind="bar", ax=ax, color="tab:green")
        ax.set_ylabel("% com AVC")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Figura 5 — Taxa de AVC por categoria (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-categorical-vs-target.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col in zip(axes, ["age", "avg_glucose_level"]):
        groups = [train_df.loc[train_df[TARGET] == c, col].dropna() for c in (0, 1)]
        ax.boxplot(groups, tick_labels=["Sem AVC", "Com AVC"])
        ax.set_title(col)
    fig.suptitle("Figura 6 — Numéricas por classe do alvo (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-numeric-vs-target.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Etapa 4 — Pré-processamento
# --------------------------------------------------------------------------- #

def cap_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Winsoriza pelos limites de IQR*1.5 calculados só no treino."""
    train_df, test_df = train_df.copy(), test_df.copy()
    affected = 0
    for col in cols:
        q1, q3 = train_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        affected += int(((train_df[col] < low) | (train_df[col] > high)).sum())
        train_df[col] = train_df[col].clip(low, high)
        test_df[col] = test_df[col].clip(low, high)
    return train_df, test_df, affected


def step4_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== Etapa 4 — Pré-processamento ===")

    train_df, test_df, n_outliers = cap_outliers(train_df, test_df, ["avg_glucose_level", "bmi"])
    print(f"Linhas de treino afetadas pelo cap de outliers (IQR*1.5, avg_glucose_level/bmi): "
          f"{n_outliers}")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    feature_names = preprocessor.get_feature_names_out()

    print(f"Shape treino após o pipeline: {X_train.shape}")
    print(f"Shape teste após o pipeline:  {X_test.shape}")
    print(f"NaN no treino: {np.isnan(X_train).sum()} | NaN no teste: {np.isnan(X_test).sum()}")
    print(f"Nomes das features: {list(feature_names)}")

    y_train = train_df[TARGET].to_numpy()

    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(X_train)
    explained = pca.explained_variance_ratio_
    print(f"\nPCA — variância explicada: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}, "
          f"soma={explained[:2].sum():.4f}")
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=["PC1", "PC2"])
    print("Loadings (|maior| por componente):")
    print(loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index).head(5))

    # Amostra para t-SNE/UMAP (mais rápido, instrui o enunciado para datasets grandes)
    sample_idx = RNG.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
    X_sample = X_train[sample_idx]
    y_sample = y_train[sample_idx]

    tsne_perplexities = [10, 40]
    tsne_projs = {
        p: TSNE(n_components=2, perplexity=p, random_state=42, init="pca").fit_transform(X_sample)
        for p in tsne_perplexities
    }

    umap_neighbors = [10, 40]
    if UMAP is not None:
        umap_projs = {
            n: UMAP(n_components=2, n_neighbors=n, random_state=42).fit_transform(X_sample)
            for n in umap_neighbors
        }
    else:
        print("\n[AVISO] umap-learn indisponível neste ambiente (numba bloqueado pelo "
              "Windows) — Figuras 7/8 saem sem o painel de UMAP. Rode este script numa "
              "máquina com umap-learn funcionando para completá-las.")
        umap_projs = None

    n_panels = 3 if umap_projs is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5))

    for c, label in ((0, "Sem AVC"), (1, "Com AVC")):
        axes[0].scatter(*pca_proj[y_train == c].T, s=6, alpha=0.5, label=label)
    axes[0].set_title("PCA")
    axes[0].legend()

    for c, label in ((0, "Sem AVC"), (1, "Com AVC")):
        axes[1].scatter(*tsne_projs[tsne_perplexities[0]][y_sample == c].T, s=6, alpha=0.5, label=label)
    axes[1].set_title(f"t-SNE (perplexity={tsne_perplexities[0]})")
    axes[1].legend()

    if umap_projs is not None:
        for c, label in ((0, "Sem AVC"), (1, "Com AVC")):
            axes[2].scatter(*umap_projs[umap_neighbors[0]][y_sample == c].T, s=6, alpha=0.5, label=label)
        axes[2].set_title(f"UMAP (n_neighbors={umap_neighbors[0]})")
        axes[2].legend()

    fig.suptitle("Figura 7 — Projeções 2D: PCA vs. t-SNE" + (" vs. UMAP (treino)" if umap_projs else " (treino)"))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig07-pca-tsne-umap.png", dpi=150)
    plt.close(fig)

    n_rows = 2 if umap_projs is not None else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows), squeeze=False)
    for ax, p in zip(axes[0], tsne_perplexities):
        for c, label in ((0, "Sem AVC"), (1, "Com AVC")):
            ax.scatter(*tsne_projs[p][y_sample == c].T, s=6, alpha=0.5, label=label)
        ax.set_title(f"t-SNE (perplexity={p})")
        ax.legend()
    if umap_projs is not None:
        for ax, n in zip(axes[1], umap_neighbors):
            for c, label in ((0, "Sem AVC"), (1, "Com AVC")):
                ax.scatter(*umap_projs[n][y_sample == c].T, s=6, alpha=0.5, label=label)
            ax.set_title(f"UMAP (n_neighbors={n})")
            ax.legend()
    fig.suptitle("Figura 8 — t-SNE" + (" e UMAP" if umap_projs else "") + " em dois parâmetros cada (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig08-tsne-umap-params.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train_df, test_df = step1_inspection(df)
    step2_univariate(train_df)
    step3_bivariate(train_df)
    step4_preprocessing(train_df, test_df)


if __name__ == "__main__":
    main()
