"""Entrega 2 — Classificação. Roda as etapas 0 a 6 e imprime todos os números do relatório.

Uso (a partir da raiz do repositório):

    python docs/projects/classification/code/run_classification.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                              precision_recall_curve, precision_recall_fscore_support,
                              roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_splits  # noqa: E402
from mlp import MLP, n_params, predict_proba, train_with_early_stopping  # noqa: E402

FIGURES = HERE.parents[0] / "figures"
CKPT_DIR = HERE / "data"

# --- Etapa 0 — definidos ANTES de treinar qualquer coisa -------------------
PRINCIPAL_METRIC = "PR-AUC (average precision)"
SUCCESS_CRITERION_FACTOR = 1.20  # PR-AUC da rede >= 1,20 x PR-AUC do melhor baseline (validação)


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "threshold": threshold,
        "precision": precision, "recall": recall, "f1": f1,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    d = load_splits()
    X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
    feature_names = d["feature_names"]
    n_features = X_train.shape[1]
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    print("=== Etapa 0 — Ponto de partida ===")
    print(f"Métrica principal: {PRINCIPAL_METRIC}")
    print(f"Critério de sucesso: PR-AUC da rede >= {SUCCESS_CRITERION_FACTOR} x PR-AUC do melhor baseline (validação)")
    print(f"Treino={X_train.shape}, Validação={X_val.shape}, Teste={X_test.shape}, pos_weight={pos_weight:.3f}")

    # === Etapa 1 — Baselines ===================================================
    print("\n=== Etapa 1 — Baselines (validação) ===")
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_val_prob = dummy.predict_proba(X_val)[:, 1]
    dummy_metrics = eval_metrics(y_val, dummy_val_prob)
    print(f"Dummy (classe majoritária): PR-AUC={dummy_metrics['pr_auc']:.4f}, "
          f"ROC-AUC={dummy_metrics['roc_auc']:.4f}, F1-macro={dummy_metrics['f1_macro']:.4f}")

    logreg = LogisticRegression(max_iter=2000, random_state=42).fit(X_train, y_train)
    logreg_val_prob = logreg.predict_proba(X_val)[:, 1]
    logreg_metrics = eval_metrics(y_val, logreg_val_prob)
    print(f"Logística: PR-AUC={logreg_metrics['pr_auc']:.4f}, ROC-AUC={logreg_metrics['roc_auc']:.4f}, "
          f"F1-macro={logreg_metrics['f1_macro']:.4f}")

    best_baseline_name = "Logística" if logreg_metrics["pr_auc"] >= dummy_metrics["pr_auc"] else "Dummy"
    best_baseline_pr_auc = max(logreg_metrics["pr_auc"], dummy_metrics["pr_auc"])
    print(f"Melhor baseline (validação): {best_baseline_name}, PR-AUC={best_baseline_pr_auc:.4f}")

    # === Etapa 2/3 — A rede (arquitetura base) e treinamento ===================
    print("\n=== Etapa 2/3 — Rede base e treinamento ===")
    base_hidden = [32, 16]
    model = MLP(n_features, hidden=base_hidden)
    print(f"Arquitetura: {n_features} -> {' -> '.join(map(str, base_hidden))} -> 1, "
          f"parâmetros={n_params(model)}, linhas de treino={len(X_train)}")

    history = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val, lr=1e-3, weight_decay=0.0,
        batch_size=64, max_epochs=300, patience=20, pos_weight=pos_weight,
        checkpoint_path=CKPT_DIR / "base.pt", seed=42,
    )
    print(f"Parou na época {history['stopped_epoch']}, "
          f"perda final treino={history['train_loss'][-1]:.4f}, val={history['val_loss'][-1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="Treino")
    ax.plot(history["val_loss"], label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("BCE (com pos_weight)")
    ax.set_title("Figura 1 — Curvas de perda (rede base)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-loss-curves.png", dpi=150)
    plt.close(fig)

    base_val_prob = predict_proba(model, X_val)
    base_val_metrics = eval_metrics(y_val, base_val_prob)
    print(f"Rede base (validação): PR-AUC={base_val_metrics['pr_auc']:.4f}, "
          f"ROC-AUC={base_val_metrics['roc_auc']:.4f}")

    # === Etapa 4 — Regularização e busca de hiperparâmetros =====================
    print("\n=== Etapa 4 — Regularização ===")
    reg_variants = {
        "sem regularização": dict(hidden=base_hidden, lr=1e-3, wd=0.0, dropout=0.0),
        "weight decay (wd=1e-3)": dict(hidden=base_hidden, lr=1e-3, wd=1e-3, dropout=0.0),
        "dropout (p=0.3)": dict(hidden=base_hidden, lr=1e-3, wd=0.0, dropout=0.3),
    }
    reg_results = {}
    for name, cfg in reg_variants.items():
        m = MLP(n_features, hidden=cfg["hidden"], dropout=cfg["dropout"])
        h = train_with_early_stopping(
            m, X_train, y_train, X_val, y_val, lr=cfg["lr"], weight_decay=cfg["wd"],
            pos_weight=pos_weight, checkpoint_path=CKPT_DIR / "reg_tmp.pt", seed=42,
        )
        prob = predict_proba(m, X_val)
        metrics = eval_metrics(y_val, prob)
        reg_results[name] = metrics["pr_auc"]
        print(f"{name}: PR-AUC(val)={metrics['pr_auc']:.4f}, época de parada={h['stopped_epoch']}")

    print("\n=== Etapa 4 — Busca de hiperparâmetros (>= 8 configurações) ===")
    search_space = [
        dict(hidden=[16], lr=1e-3, wd=0.0, dropout=0.0),
        dict(hidden=[32], lr=1e-3, wd=0.0, dropout=0.0),
        dict(hidden=[32, 16], lr=1e-3, wd=0.0, dropout=0.0),
        dict(hidden=[32, 16], lr=1e-3, wd=1e-3, dropout=0.0),
        dict(hidden=[32, 16], lr=1e-3, wd=0.0, dropout=0.3),
        dict(hidden=[32, 16], lr=1e-3, wd=1e-4, dropout=0.2),
        dict(hidden=[64, 32], lr=1e-3, wd=1e-4, dropout=0.2),
        dict(hidden=[64, 32, 16], lr=1e-3, wd=1e-4, dropout=0.2),
        dict(hidden=[32, 16], lr=1e-2, wd=0.0, dropout=0.0),
        dict(hidden=[32, 16], lr=3e-4, wd=0.0, dropout=0.0),
    ]
    search_results = []
    for cfg in search_space:
        m = MLP(n_features, hidden=cfg["hidden"], dropout=cfg["dropout"])
        h = train_with_early_stopping(
            m, X_train, y_train, X_val, y_val, lr=cfg["lr"], weight_decay=cfg["wd"],
            pos_weight=pos_weight, checkpoint_path=CKPT_DIR / "search_tmp.pt", seed=42,
        )
        prob = predict_proba(m, X_val)
        metrics = eval_metrics(y_val, prob)
        search_results.append({**cfg, "pr_auc_val": metrics["pr_auc"], "stopped_epoch": h["stopped_epoch"]})

    search_results.sort(key=lambda r: r["pr_auc_val"], reverse=True)
    for r in search_results:
        print(f"hidden={r['hidden']}, lr={r['lr']}, wd={r['wd']}, dropout={r['dropout']} "
              f"-> PR-AUC(val)={r['pr_auc_val']:.4f} (parou na época {r['stopped_epoch']})")

    best_cfg = search_results[0]
    print(f"\nConfiguração escolhida: {best_cfg}")

    # === Etapa 5 — Avaliação no teste (uma única vez, 3 sementes) ===============
    print("\n=== Etapa 5 — Avaliação no teste (3 sementes) ===")
    seeds = [42, 43, 44]
    test_metrics_per_seed = []
    final_model_seed42 = None
    for seed in seeds:
        m = MLP(n_features, hidden=best_cfg["hidden"], dropout=best_cfg["dropout"])
        train_with_early_stopping(
            m, X_train, y_train, X_val, y_val, lr=best_cfg["lr"], weight_decay=best_cfg["wd"],
            pos_weight=pos_weight, checkpoint_path=CKPT_DIR / f"final_seed{seed}.pt", seed=seed,
        )
        val_prob = predict_proba(m, X_val)
        threshold = best_threshold_by_f1(y_val, val_prob)
        test_prob = predict_proba(m, X_test)
        metrics = eval_metrics(y_test, test_prob, threshold=threshold)
        test_metrics_per_seed.append(metrics)
        print(f"seed={seed}: limiar={threshold:.3f}, PR-AUC(teste)={metrics['pr_auc']:.4f}, "
              f"ROC-AUC(teste)={metrics['roc_auc']:.4f}, F1-macro(teste)={metrics['f1_macro']:.4f}")
        if seed == 42:
            final_model_seed42 = m
            final_test_prob_seed42 = test_prob
            final_threshold_seed42 = threshold

    pr_aucs = np.array([m["pr_auc"] for m in test_metrics_per_seed])
    print(f"\nPR-AUC no teste (média ± desvio, {len(seeds)} sementes): "
          f"{pr_aucs.mean():.4f} +/- {pr_aucs.std():.4f}")

    dummy_test_prob = dummy.predict_proba(X_test)[:, 1]
    dummy_test_metrics = eval_metrics(y_test, dummy_test_prob)
    logreg_test_prob = logreg.predict_proba(X_test)[:, 1]
    logreg_test_metrics = eval_metrics(y_test, logreg_test_prob)
    print(f"Dummy (teste): PR-AUC={dummy_test_metrics['pr_auc']:.4f}")
    print(f"Logística (teste): PR-AUC={logreg_test_metrics['pr_auc']:.4f}")

    best_baseline_test_pr_auc = max(dummy_test_metrics["pr_auc"], logreg_test_metrics["pr_auc"])
    criterion_met = pr_aucs.mean() >= SUCCESS_CRITERION_FACTOR * best_baseline_test_pr_auc
    print(f"Critério de sucesso atingido? {'SIM' if criterion_met else 'NÃO'} "
          f"({pr_aucs.mean():.4f} vs. {SUCCESS_CRITERION_FACTOR}x{best_baseline_test_pr_auc:.4f}="
          f"{SUCCESS_CRITERION_FACTOR * best_baseline_test_pr_auc:.4f})")

    cm = confusion_matrix(y_test, (final_test_prob_seed42 >= final_threshold_seed42).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sem AVC", "Com AVC"]); ax.set_yticklabels(["Sem AVC", "Com AVC"])
    ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    ax.set_title("Figura 2 — Matriz de confusão (teste, seed=42)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-confusion-matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, final_test_prob_seed42)
    prec, rec, _ = precision_recall_curve(y_test, final_test_prob_seed42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("Taxa de falso positivo")
    axes[0].set_ylabel("Taxa de verdadeiro positivo")
    axes[0].set_title("ROC (teste, seed=42)")
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precisão")
    axes[1].set_title("Precisão-Recall (teste, seed=42)")
    fig.suptitle("Figura 3 — Curvas ROC e Precisão-Recall")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-roc-pr-curves.png", dpi=150)
    plt.close(fig)

    # === Etapa 6 — Análise de erros =============================================
    print("\n=== Etapa 6 — Análise de erros (teste, seed=42) ===")
    test_df = d["test_df"].reset_index(drop=True)
    y_pred_final = (final_test_prob_seed42 >= final_threshold_seed42).astype(int)
    wrong = y_pred_final != y_test

    for subgroup_col in ["hypertension", "heart_disease"]:
        print(f"\nTaxa de erro por '{subgroup_col}':")
        for value in sorted(test_df[subgroup_col].unique()):
            mask = (test_df[subgroup_col] == value).to_numpy()
            err_rate = wrong[mask].mean()
            print(f"  {subgroup_col}={value}: erro={err_rate:.4f} (n={mask.sum()})")

    worst_idx = np.argsort(-np.abs(final_test_prob_seed42 - y_test))[:5]
    print("\n5 piores exemplos (maior |probabilidade - rótulo|):")
    for i in worst_idx:
        print(f"  idx={i}, y={int(y_test[i])}, prob={final_test_prob_seed42[i]:.4f}, "
              f"age={test_df.loc[i, 'age']}, hypertension={test_df.loc[i, 'hypertension']}, "
              f"heart_disease={test_df.loc[i, 'heart_disease']}")

    print("\nImportância por permutação (PR-AUC no teste, seed=42):")
    rng = np.random.default_rng(42)
    base_pr_auc = average_precision_score(y_test, final_test_prob_seed42)
    importances = []
    for j, fname in enumerate(feature_names):
        X_perm = X_test.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_prob = predict_proba(final_model_seed42, X_perm)
        drop = base_pr_auc - average_precision_score(y_test, perm_prob)
        importances.append((fname, drop))
    importances.sort(key=lambda t: t[1], reverse=True)
    for fname, drop in importances[:8]:
        print(f"  {fname}: queda de PR-AUC = {drop:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    top = importances[:8]
    ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1])
    ax.set_xlabel("Queda de PR-AUC ao embaralhar a feature")
    ax.set_title("Figura 4 — Importância por permutação (top 8, teste)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-permutation-importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
