"""
main.py
=======
Master execution pipeline for the Telco Customer Churn Case Study:
Department of Artificial Intelligence and Data Science,
KPR Institute of Engineering and Technology (Course Code: U21ADP05).

Executes:
1. Data Acquisition, Preprocessing & Stratified Splitting
2. Ben Fry 7-Stage Exploratory Data Visualizations (6 Figures)
3. Deep Learning MLP Training, Hyperparameter Tuning & Permutation Importance
4. Model Evaluation Visualizations (4 Figures)
5. Comprehensive 11-Section Project Report (Project_Report.pdf)
"""

import os
import sys
import time

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import acquire_dataset, clean_and_preprocess
from visualization import generate_all_visualizations
from dl_model import (
    tune_hyperparameters,
    evaluate_model_on_test,
    compute_permutation_importance,
    plot_step5_evaluation_visualizations
)
from generate_pdf_report import build_pdf_report


def main():
    print("=" * 75)
    print(" KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY - DEPT. OF AI & DATA SCIENCE")
    print(" U21ADP05 - Exploratory Data Analysis and Visualization (Assignment II)")
    print(" Case Study: Exploratory Data Analysis & Deep Learning for Customer Churn")
    print("=" * 75)
    start_time = time.time()

    # Step 1 & 2: Data Acquisition & Preprocessing
    print("\n>>> STEP 1 & 2: Ingesting & Preprocessing Dataset...")
    df_raw = acquire_dataset()
    data = clean_and_preprocess(df_raw)
    meta = data["metadata"]
    print(f" -> Raw Shape: {meta['raw_shape']} | Clean Shape: {meta['clean_shape']}")
    print(f" -> Train: {meta['train_samples']} | Val: {meta['val_samples']} | Test: {meta['test_samples']}")
    print(f" -> Features: {meta['num_features']} | Churn Rate: {meta['target_balance']['Churn_Rate']:.2f}%")

    # Step 3: Exploratory Data Visualization (Fry's 7 Stages)
    print("\n>>> STEP 3: Generating Exploratory Visualizations (Ben Fry's 7 Stages)...")
    vis_results = generate_all_visualizations(data["df_clean"])
    print(f" -> Successfully generated {len(vis_results)} exploratory figures in figures/")

    # Step 4: Model Building & Hyperparameter Tuning
    print("\n>>> STEP 4: Deep Learning Model Building & Hyperparameter Tuning...")
    best_model, best_hist, tuning_df = tune_hyperparameters(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    )
    print(f" -> Hyperparameter tuning completed across 3 candidate architectures.")

    # Step 5: Model Evaluation on Unseen Test Partition
    print("\n>>> STEP 5: Model Evaluation & Holdout Validation...")
    eval_res = evaluate_model_on_test(
        best_model, data["X_test"], data["y_test"], data["X_train"], data["y_train"]
    )
    dl_m = eval_res["dl_metrics"]
    lr_m = eval_res["lr_metrics"]
    print(f" -> Deep Learning Test Accuracy: {dl_m['Accuracy']*100:.2f}% | ROC-AUC: {dl_m['ROC_AUC']:.4f}")
    print(f" -> Baseline (Logistic Reg) Accuracy: {lr_m['Accuracy']*100:.2f}% | ROC-AUC: {lr_m['ROC_AUC']:.4f}")

    # Permutation Feature Importance
    print("\n>>> Computing Permutation Feature Importance...")
    feat_imp = compute_permutation_importance(
        best_model, data["X_test"], data["y_test"], meta["feature_names"]
    )
    print(" -> Top 5 Impactful Features:")
    for idx, r in feat_imp.head(5).iterrows():
        print(f"    {r['Feature']:30s} Drop in AUC: {r['Importance']:.5f}")

    # Generate Evaluation Charts
    print("\n>>> Generating Step 5 Model Evaluation Visualizations...")
    eval_figs = plot_step5_evaluation_visualizations(best_hist, eval_res, data["y_test"])
    print(f" -> Generated {len(eval_figs)} model evaluation figures in figures/")

    # Step 6 & 7: Comprehensive PDF Report Compilation
    print("\n>>> STEP 6 & 7: Generating Formal 11-Section Project Report (PDF)...")
    pdf_path = build_pdf_report(meta, tuning_df, eval_res, feat_imp)
    print(f" -> Publication Report generated: {pdf_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f" SUCCESS: Full Pipeline Executed in {elapsed:.1f} seconds!")
    print(" Deliverables Ready:")
    print(f"  1. Dataset:                  dataset/telco_customer_churn.csv")
    print(f"  2. Figures (10 High-Res):    figures/ (01 to 10)")
    print(f"  3. Pipeline Script:          main.py & src/")
    print(f"  4. Publication PDF Report:   {pdf_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
