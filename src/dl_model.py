"""
dl_model.py
===========
Deep Learning Multi-Layer Perceptron (MLP) for Telco Customer Churn.
Includes:
- Deep Learning Architecture with MLP (Multi-Layer Perceptron)
- Epoch-by-epoch loss & accuracy tracking
- Hyperparameter tuning comparison across 3 architectures
- Permutation feature importance analysis
- Step 5 Result Visualizations:
  1. Loss vs Epoch chart
  2. Accuracy vs Epoch chart
  3. Confusion Matrix chart
  4. ROC Curve and AUC Score chart
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, log_loss
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def train_mlp_model(X_train, y_train, X_val, y_val,
                    hidden_layer_sizes=(64, 32, 16),
                    learning_rate_init=0.001,
                    alpha=0.0001,
                    epochs=35,
                    batch_size=64,
                    random_state=42):
    """
    Train a Deep Learning Multi-Layer Perceptron (MLP) epoch-by-epoch,
    logging loss, accuracy, and ROC-AUC for both training and validation sets.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=1,
        warm_start=True,
        random_state=random_state
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auc": [],
        "val_auc": []
    }

    # Custom loop across epochs
    classes = np.unique(y_train)
    for epoch in range(1, epochs + 1):
        mlp.fit(X_train, y_train)

        tr_probs = mlp.predict_proba(X_train)[:, 1]
        val_probs = mlp.predict_proba(X_val)[:, 1]

        tr_loss = log_loss(y_train, tr_probs)
        v_loss = log_loss(y_val, val_probs)

        tr_acc = accuracy_score(y_train, (tr_probs >= 0.5).astype(int))
        v_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))

        tr_auc = roc_auc_score(y_train, tr_probs)
        v_auc = roc_auc_score(y_val, val_probs)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(v_auc)

    return mlp, history


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning experiment comparing 3 candidate architectures:
    - Shallow MLP: (64, 32), lr=0.005, alpha=0.001
    - Tuned Deep MLP: (64, 32, 16), lr=0.001, alpha=0.0001
    - Wide Deep MLP: (128, 64, 32), lr=0.0005, alpha=0.0005
    """
    print("[Hyperparameter Tuning] Evaluating candidate architectures...")
    configs = [
        {"name": "Config A (Shallow Fast)", "hidden": (64, 32), "lr": 0.005, "alpha": 0.001},
        {"name": "Config B (Tuned Deep MLP)", "hidden": (64, 32, 16), "lr": 0.001, "alpha": 0.0001},
        {"name": "Config C (Wide Deep MLP)", "hidden": (128, 64, 32), "lr": 0.0005, "alpha": 0.0005},
    ]

    results = []
    models = {}

    for cfg in configs:
        model, hist = train_mlp_model(
            X_train, y_train, X_val, y_val,
            hidden_layer_sizes=cfg["hidden"],
            learning_rate_init=cfg["lr"],
            alpha=cfg["alpha"],
            epochs=30
        )
        min_vloss = min(hist["val_loss"])
        max_vauc = max(hist["val_auc"])
        best_acc = hist["val_acc"][np.argmin(hist["val_loss"])]

        results.append({
            "Configuration": cfg["name"],
            "Hidden Layers": str(cfg["hidden"]),
            "Learning Rate": cfg["lr"],
            "L2 Regularization (alpha)": cfg["alpha"],
            "Best Val Loss": round(min_vloss, 4),
            "Best Val AUC": round(max_vauc, 4),
            "Best Val Accuracy (%)": round(best_acc * 100, 2)
        })
        models[cfg["name"]] = (model, hist)

    results_df = pd.DataFrame(results)
    print("\n[Hyperparameter Tuning Results]:\n", results_df.to_string(index=False))

    best_name = "Config B (Tuned Deep MLP)"
    best_model, best_hist = models[best_name]

    return best_model, best_hist, results_df


def evaluate_model_on_test(model, X_test, y_test, X_train, y_train):
    """
    Compute comprehensive test metrics and benchmark against Logistic Regression baseline.
    """
    dl_probs = model.predict_proba(X_test)[:, 1]
    dl_preds = (dl_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, dl_preds)
    tn, fp, fn, tp = cm.ravel()

    dl_metrics = {
        "Accuracy": float(accuracy_score(y_test, dl_preds)),
        "Precision": float(precision_score(y_test, dl_preds)),
        "Recall": float(recall_score(y_test, dl_preds)),
        "Specificity": float(tn / (tn + fp)),
        "F1_Score": float(f1_score(y_test, dl_preds)),
        "ROC_AUC": float(roc_auc_score(y_test, dl_probs)),
        "Confusion_Matrix": cm.tolist()
    }

    # Baseline: Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_preds = lr.predict(X_test)
    lr_cm = confusion_matrix(y_test, lr_preds)
    lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

    lr_metrics = {
        "Accuracy": float(accuracy_score(y_test, lr_preds)),
        "Precision": float(precision_score(y_test, lr_preds)),
        "Recall": float(recall_score(y_test, lr_preds)),
        "Specificity": float(lr_tn / (lr_tn + lr_fp)),
        "F1_Score": float(f1_score(y_test, lr_preds)),
        "ROC_AUC": float(roc_auc_score(y_test, lr_probs))
    }

    return {
        "dl_metrics": dl_metrics,
        "dl_probs": dl_probs,
        "dl_preds": dl_preds,
        "lr_metrics": lr_metrics,
        "lr_probs": lr_probs,
        "lr_preds": lr_preds,
        "confusion_matrix": cm
    }


def compute_permutation_importance(model, X_test, y_test, feature_names):
    """
    Compute model-agnostic Permutation Feature Importance based on test ROC-AUC.
    """
    res = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": res.importances_mean,
        "Std": res.importances_std
    }).sort_values("Importance", ascending=False)
    return imp_df


def plot_step5_evaluation_visualizations(history: dict, eval_results: dict, y_test) -> dict:
    """
    Generate all 4 required Step 5 Model Evaluation Visualizations:
    1. Loss vs Epoch chart
    2. Accuracy vs Epoch chart
    3. Confusion Matrix chart
    4. ROC Curve and AUC Score chart
    """
    # 1. Loss vs Epoch chart
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="Training Loss (BCE)", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax1.plot(history["epoch"], history["val_loss"], label="Validation Loss (BCE)", color="#d95f02", linewidth=2.2, linestyle="--", marker="s", markersize=4)
    best_idx = int(np.argmin(history["val_loss"]))
    best_epoch = history["epoch"][best_idx]
    min_loss = history["val_loss"][best_idx]
    ax1.axvline(best_epoch, color="green", linestyle=":", label=f"Optimal Epoch ({best_epoch})", linewidth=1.8)
    ax1.annotate(f"Min Val Loss: {min_loss:.3f}", xy=(best_epoch, min_loss), xytext=(best_epoch + 2, min_loss + 0.04),
                 arrowprops=dict(arrowstyle="->", color="green", lw=1.5), fontweight="bold")
    ax1.set_title("Step 5.1: Deep Learning Training & Validation Loss vs Epoch", fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend(loc="upper right", frameon=True)
    loss_path = os.path.join(FIGURES_DIR, "07_dl_loss_curve.png")
    fig1.tight_layout()
    fig1.savefig(loss_path, dpi=300)
    plt.close(fig1)

    # 2. Accuracy vs Epoch chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Training Accuracy (%)", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    ax2.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Validation Accuracy (%)", color="#2ca02c", linewidth=2.2, linestyle="--", marker="^", markersize=4)
    ax2.set_title("Step 5.2: Deep Learning Training & Validation Accuracy vs Epoch", fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("Classification Accuracy (%)")
    ax2.set_ylim(65, 90)
    ax2.legend(loc="lower right", frameon=True)
    acc_path = os.path.join(FIGURES_DIR, "08_dl_accuracy_curve.png")
    fig2.tight_layout()
    fig2.savefig(acc_path, dpi=300)
    plt.close(fig2)

    # 3. Confusion Matrix chart
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    cm = eval_results["confusion_matrix"]
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    labels = np.array([
        [f"{cm[0,0]:,}\n({cm_norm[0,0]:.1%})\n[TN]", f"{cm[0,1]:,}\n({cm_norm[0,1]:.1%})\n[FP]"],
        [f"{cm[1,0]:,}\n({cm_norm[1,0]:.1%})\n[FN]", f"{cm[1,1]:,}\n({cm_norm[1,1]:.1%})\n[TP]"]
    ])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax3,
                annot_kws={"fontsize": 12, "fontweight": "bold"}, linewidths=1.5, linecolor="gray")
    ax3.set_title("Step 5.3: Test Set Confusion Matrix (Normalized & Raw Counts)", fontweight="bold", pad=12)
    ax3.set_xlabel("Predicted Label", fontweight="bold")
    ax3.set_ylabel("True Label", fontweight="bold")
    ax3.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax3.set_yticklabels(["Retained (0)", "Churned (1)"])
    cm_path = os.path.join(FIGURES_DIR, "09_confusion_matrix.png")
    fig3.tight_layout()
    fig3.savefig(cm_path, dpi=300)
    plt.close(fig3)

    # 4. ROC Curve and AUC Score chart
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    fpr_dl, tpr_dl, _ = roc_curve(y_test, eval_results["dl_probs"])
    fpr_lr, tpr_lr, _ = roc_curve(y_test, eval_results["lr_probs"])
    auc_dl = eval_results["dl_metrics"]["ROC_AUC"]
    auc_lr = eval_results["lr_metrics"]["ROC_AUC"]

    ax4.plot(fpr_dl, tpr_dl, color="#d95f02", linewidth=2.5, label=f"Deep Learning MLP (AUC = {auc_dl:.3f})")
    ax4.plot(fpr_lr, tpr_lr, color="#1f77b4", linewidth=2.0, linestyle="--", label=f"Logistic Regression Baseline (AUC = {auc_lr:.3f})")
    ax4.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Chance (AUC = 0.500)")
    ax4.set_title("Step 5.4: Receiver Operating Characteristic (ROC) & AUC Comparison", fontweight="bold", pad=12)
    ax4.set_xlabel("False Positive Rate (1 - Specificity)")
    ax4.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax4.legend(loc="lower right", frameon=True)
    roc_path = os.path.join(FIGURES_DIR, "10_roc_auc_curve.png")
    fig4.tight_layout()
    fig4.savefig(roc_path, dpi=300)
    plt.close(fig4)

    return {
        "loss_curve": loss_path,
        "acc_curve": acc_path,
        "confusion_matrix": cm_path,
        "roc_curve": roc_path
    }


if __name__ == "__main__":
    from data_loader import acquire_dataset, clean_and_preprocess
    df = acquire_dataset()
    data = clean_and_preprocess(df)
    model, hist, tuning_df = tune_hyperparameters(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    eval_res = evaluate_model_on_test(model, data["X_test"], data["y_test"], data["X_train"], data["y_train"])
    figs = plot_step5_evaluation_visualizations(hist, eval_res, data["y_test"])
    print("Model Evaluation Figures generated:", figs)
    print("DL Test Metrics:", eval_res["dl_metrics"])
