"""
Telco Customer Churn - EDA, Visualization and Deep Learning (MLP)
U21ADP05 - Exploratory Data Analysis and Visualization | KPRIET
"""
import os, copy, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, log_loss)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("figures", exist_ok=True)
sns.set_style("whitegrid")
PAL = {"No": "#2B5F91", "Yes": "#D95F02"}

# ---------------- Stages 1-2: Acquire & Parse ----------------
df = pd.read_csv("dataset/telco_customer_churn.csv")
print("Shape:", df.shape)
print(df.dtypes)

# TotalCharges is read as text because 11 rows hold a blank string " "
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan),
                                   errors="coerce")
blank = df["TotalCharges"].isna()
print("Blank TotalCharges:", blank.sum(), "| their tenure:",
      df.loc[blank, "tenure"].unique())      # all tenure == 0 -> new customers
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)   # zero billing cycles

# ---------------- Stage 3: Filter / data quality ----------------
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])

cont_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
for c in cont_cols:                                   # Tukey IQR fences
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    print(f"{c}: IQR={iqr:.2f}  bounds=({lo:.2f}, {hi:.2f})  outliers={n_out}")

# ---------------- Stage 4: Mine - descriptive statistics ----------------
summary = df[cont_cols].describe().T
summary["skewness"] = df[cont_cols].skew()
summary["kurtosis"] = df[cont_cols].kurt()
print(summary.round(2))
print(df["Churn"].value_counts(), "\n", df["Churn"].value_counts(normalize=True).round(3))

# ---------------- Encoding ----------------
df["Churn_Num"] = (df["Churn"] == "Yes").astype(int)
data = df.drop(columns=["customerID", "Churn"]).copy()
data["gender"] = (data["gender"] == "Male").astype(int)
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    data[c] = (data[c] == "Yes").astype(int)
y = data.pop("Churn_Num").values
X = pd.get_dummies(data, drop_first=True).astype(float)   # 10 multi-class cols
print("Model input dimensions:", X.shape[1])               # expected: 30

# ---------------- Stratified 70 / 15 / 15 split ----------------
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=SEED)
X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
for name, part in [("train", y_train), ("val", y_val), ("test", y_test)]:
    print(f"{name}: {len(part)} rows | retained={np.sum(part == 0)} churned={np.sum(part == 1)}")

# ---------------- Standardization (fit on TRAIN only) ----------------
scaler = StandardScaler()
X_train[cont_cols] = scaler.fit_transform(X_train[cont_cols])
X_val[cont_cols] = scaler.transform(X_val[cont_cols])
X_test[cont_cols] = scaler.transform(X_test[cont_cols])
# ======================= VISUALIZATION (Stages 5-7) =======================
# Figure 1 - discrete distribution of the target
counts = df["Churn"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].bar(["Retained (No)", "Churned (Yes)"], [counts["No"], counts["Yes"]],
          color=[PAL["No"], PAL["Yes"]], edgecolor="black")
for i, v in enumerate([counts["No"], counts["Yes"]]):
    ax[0].text(i, v / 2, f"{v:,}\n({v / len(df):.1%})", ha="center",
               color="white", fontweight="bold")
ax[0].set(title="Discrete Class Distribution", ylabel="Customer Count")
ax[1].pie([counts["No"], counts["Yes"]], labels=["Retained (No)", "Churned (Yes)"],
          colors=[PAL["No"], PAL["Yes"]], autopct="%1.1f%%", startangle=90,
          pctdistance=0.8, wedgeprops=dict(width=0.4, edgecolor="black"))
ax[1].set_title("Proportion of Churn vs Retention")
fig.suptitle("Target Class Distribution Analysis", fontweight="bold")
fig.savefig("figures/01_target_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2 - continuous distributions (histogram + KDE, mean/median lines)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, c, col in zip(axes, cont_cols, ["#1f77b4", "#2ca02c", "#9467bd"]):
    sns.histplot(df[c], bins=30, stat="density", kde=True, color=col, ax=ax)
    ax.axvline(df[c].mean(), color="red", ls="--", label=f"Mean: {df[c].mean():.1f}")
    ax.axvline(df[c].median(), color="navy", ls=":", label=f"Median: {df[c].median():.1f}")
    ax.set_title(f"{c}\nSkewness: {df[c].skew():.2f} | Kurtosis: {df[c].kurt():.2f}",
                 fontweight="bold")
    ax.legend()
fig.suptitle("Continuous Probability Density Functions (KDE & Histograms)", fontweight="bold")
fig.savefig("figures/02_univariate_continuous_kde.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 3 - boxplots by churn + Mann-Whitney U test
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, c in zip(axes, cont_cols):
    a = df.loc[df["Churn"] == "No", c]
    b = df.loc[df["Churn"] == "Yes", c]
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    sns.boxplot(data=df, x="Churn", y=c, palette=PAL, ax=ax, width=0.5)
    sns.stripplot(data=df, x="Churn", y=c, color="black", alpha=0.08, size=2, ax=ax)
    ax.set_title(f"{c} by Churn Status\nMW-U p-value: {p:.2e}", fontweight="bold")
    ax.text(0.5, 0.95, f"Median No: {a.median():.1f}\nMedian Yes: {b.median():.1f}",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round", fc="lightyellow"))
fig.savefig("figures/03_bivariate_boxplots.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 4 - contract type vs churn + Chi-square test of independence
ct = pd.crosstab(df["Contract"], df["Churn"])
chi2, p, dof, _ = stats.chi2_contingency(ct)
pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ct.plot(kind="bar", color=[PAL["No"], PAL["Yes"]], edgecolor="black", rot=0, ax=ax[0])
ax[0].set(title="Customer Counts by Contract Type and Churn", ylabel="Number of Customers")
pct.plot(kind="bar", stacked=True, color=[PAL["No"], PAL["Yes"]], edgecolor="black",
         rot=0, ax=ax[1])
ax[1].set(title=f"Churn Rate per Contract\nChi-Square: {chi2:.1f} (p = {p:.1e})",
          ylabel="Percentage (%)")
for i, v in enumerate(pct["Yes"]):
    ax[1].text(i, 50, f"Churn: {v:.1f}%", ha="center", color="white", fontweight="bold")
fig.savefig("figures/04_contract_churn_mosaic.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 5 - Pearson correlation heatmap (lower triangle)
corr_df = pd.DataFrame({
    "Tenure": df["tenure"], "Monthly Charges": df["MonthlyCharges"],
    "Total Charges": df["TotalCharges"], "Senior Citizen": df["SeniorCitizen"],
    "Partner": (df["Partner"] == "Yes").astype(int),
    "Dependents": (df["Dependents"] == "Yes").astype(int),
    "Paperless Billing": (df["PaperlessBilling"] == "Yes").astype(int),
    "Month-to-Month": (df["Contract"] == "Month-to-month").astype(int),
    "Two-Year": (df["Contract"] == "Two year").astype(int),
    "Churn": df["Churn_Num"]})
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(11, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", vmin=-0.8, vmax=0.8,
            cmap=sns.diverging_palette(220, 10, as_cmap=True), linewidths=0.5,
            cbar_kws={"shrink": 0.7})
plt.title("Multivariate Pearson Correlation Matrix", fontweight="bold")
plt.grid(False)
plt.savefig("figures/05_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 6 - multivariate pairplot coloured by churn
g = sns.pairplot(df[cont_cols + ["Churn"]], hue="Churn", palette=PAL, corner=True,
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 18})
g.figure.suptitle("Multivariate Pairwise Interactions and Churn Separation",
                  y=1.02, fontweight="bold")
g.savefig("figures/06_pairplot_multivariate.png", dpi=300, bbox_inches="tight")
plt.close("all")

# Stage 7 (Interact) - interactive Plotly view, filterable by legend click / zoom
try:
    import plotly.express as px
    fig_i = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn",
                       color_discrete_map=PAL, opacity=0.6,
                       hover_data=["Contract", "InternetService", "PaymentMethod"],
                       title="Interactive: Tenure vs Monthly Charges by Churn")
    fig_i.write_html("figures/interactive_tenure_vs_charges.html")
except ImportError:
    print("plotly not installed - skipping interactive figure (pip install plotly)")
# ======================= DEEP LEARNING: MLP + TUNING =======================
def train_mlp(hidden, lr, alpha, epochs=30):
    """Train epoch-by-epoch with partial_fit so per-epoch curves are recorded.
    alpha is the L2 weight-decay penalty; the best-validation-loss epoch is kept."""
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        alpha=alpha, learning_rate_init=lr, batch_size=64,
                        random_state=SEED)
    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "val_auc": []}
    best_val, best_model, best_epoch = np.inf, None, 0
    for ep in range(1, epochs + 1):
        mlp.partial_fit(X_train, y_train, classes=np.array([0, 1]))   # one epoch
        p_tr = mlp.predict_proba(X_train)[:, 1]
        p_va = mlp.predict_proba(X_val)[:, 1]
        hist["loss"].append(log_loss(y_train, p_tr))
        hist["val_loss"].append(log_loss(y_val, p_va))
        hist["acc"].append(100 * accuracy_score(y_train, p_tr >= 0.5))
        hist["val_acc"].append(100 * accuracy_score(y_val, p_va >= 0.5))
        hist["val_auc"].append(roc_auc_score(y_val, p_va))
        if hist["val_loss"][-1] < best_val:                           # checkpoint
            best_val, best_model, best_epoch = hist["val_loss"][-1], copy.deepcopy(mlp), ep
    return best_model, hist, best_epoch

# Three candidate configurations (basic hyperparameter tuning)
configs = {
    "A (Shallow Fast)":   ((64, 32),      5e-3, 1e-3),
    "B (Tuned Deep MLP)": ((64, 32, 16),  1e-3, 1e-4),
    "C (Wide Deep MLP)":  ((128, 64, 32), 5e-4, 5e-4),
}
rows, fitted = [], {}
for name, (hidden, lr, alpha) in configs.items():
    model, hist, best_ep = train_mlp(hidden, lr, alpha)
    n_params = (sum(w.size for w in model.coefs_) +
                sum(b.size for b in model.intercepts_))
    rows.append({"Config": name, "Hidden": hidden, "LR": lr, "L2": alpha,
                 "Best epoch": best_ep, "Params": n_params,
                 "Val loss": min(hist["val_loss"]),
                 "Val AUC": hist["val_auc"][best_ep - 1],
                 "Val acc %": hist["val_acc"][best_ep - 1]})
    fitted[name] = (model, hist, best_ep)
print(pd.DataFrame(rows).round(4).to_string(index=False))

# Selected production model: Config B
mlp, hist, best_ep = fitted["B (Tuned Deep MLP)"]
epochs = range(1, len(hist["loss"]) + 1)

# Figure 7 - loss vs epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs, hist["loss"], "o-", label="Training Loss (BCE)")
plt.plot(epochs, hist["val_loss"], "s--", color="#D95F02", label="Validation Loss (BCE)")
plt.axvline(best_ep, color="green", ls=":", label=f"Optimal Epoch ({best_ep})")
plt.xlabel("Epoch Number"); plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training & Validation Loss vs Epoch", fontweight="bold", pad=12)
plt.legend(); plt.savefig("figures/07_dl_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 8 - accuracy vs epoch
plt.figure(figsize=(8, 5))
plt.plot(epochs, hist["acc"], "o-", label="Training Accuracy (%)")
plt.plot(epochs, hist["val_acc"], "^--", color="green", label="Validation Accuracy (%)")
plt.xlabel("Epoch Number"); plt.ylabel("Classification Accuracy (%)")
plt.title("Training & Validation Accuracy vs Epoch", fontweight="bold", pad=12)
plt.legend(); plt.savefig("figures/08_dl_accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.close()
# ======================= EVALUATION ON THE TEST SET =======================
def report(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred), "Specificity": tn / (tn + fp),
            "F1": f1_score(y_true, pred), "ROC-AUC": roc_auc_score(y_true, prob)}, (tn, fp, fn, tp)

prob_mlp = mlp.predict_proba(X_test)[:, 1]
# Regularized logistic-regression baseline (class_weight balances the 73.5/26.5 split)
base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X_train, y_train)
prob_lr = base.predict_proba(X_test)[:, 1]

m_mlp, (tn, fp, fn, tp) = report(y_test, prob_mlp)
m_lr, _ = report(y_test, prob_lr)
print(pd.DataFrame({"Deep Learning MLP": m_mlp, "Logistic Regression": m_lr}).round(4))

# Figure 9 - confusion matrix (normalized % and raw counts)
cm = np.array([[tn, fp], [fn, tp]])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
labels = np.array([[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)\n[{t}]"
                    for j, t in enumerate(row)] for i, row in enumerate([["TN", "FP"], ["FN", "TP"]])])
plt.figure(figsize=(7, 6))
sns.heatmap(cm_pct, annot=labels, fmt="", cmap="Blues", cbar=False, linewidths=2,
            linecolor="gray", xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"], annot_kws={"fontweight": "bold"})
plt.xlabel("Predicted Label", fontweight="bold"); plt.ylabel("True Label", fontweight="bold")
plt.title("Test Set Confusion Matrix (Normalized & Raw Counts)", fontweight="bold")
plt.savefig("figures/09_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 10 - ROC curves: MLP vs logistic regression vs random chance
plt.figure(figsize=(8, 6))
for prob, lab, st, col in [(prob_mlp, "Deep Learning MLP", "-", "#D95F02"),
                           (prob_lr, "Logistic Regression Baseline", "--", "#1f77b4")]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, st, color=col, lw=2.5, label=f"{lab} (AUC = {roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], ":", color="gray", label="Random Chance (AUC = 0.500)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title("Receiver Operating Characteristic (ROC) & AUC Comparison", fontweight="bold")
plt.legend(loc="lower right"); plt.savefig("figures/10_roc_auc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Permutation feature importance (drop in test ROC-AUC, 5 repeats)
pi = permutation_importance(mlp, X_test, y_test, scoring="roc_auc",
                            n_repeats=5, random_state=SEED)
imp = (pd.DataFrame({"Feature": X.columns, "Importance": pi.importances_mean,
                     "Std": pi.importances_std})
       .sort_values("Importance", ascending=False).head(10))
print(imp.round(5).to_string(index=False))
