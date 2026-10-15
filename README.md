# Exploratory Data Analysis, Statistical Visualization & Deep Learning on Telco Customer Churn

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Deep%20Learning-MLP-orange.svg)](https://scikit-learn.org/)
[![Visuals](https://img.shields.io/badge/Visualization-Ben%20Fry%207%20Stages-purple.svg)](https://benfry.com/)
[![Institution](https://img.shields.io/badge/Institution-KPRIET-navy.svg)](https://kpriet.ac.in/)

A comprehensive case study on customer attrition prediction applying **Ben Fry's 7 Stages of Data Visualization**, rigorous statistical hypothesis testing, and a **Deep Learning Multi-Layer Perceptron (MLP)** with hyperparameter tuning.

Submitted for **Course Code: U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II, 40 Marks)**, Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology.

---

## 📌 Executive Summary

* **Domain & Application:** Telecommunications Subscriber Retention & Revenue Optimization.
* **Dataset:** IBM Telco Customer Churn (7,043 customer accounts, 21 multi-faceted features).
* **Key Findings:**
  * Tenure and Contract Duration are the paramount drivers of retention.
  * Month-to-month contracts experience a staggering **42.7% churn rate**, compared to **11.3%** for 1-Year and **2.8%** for 2-Year contracts ($\chi^2 = 1184.6, p < 0.001$).
  * Churners display significantly higher median monthly charges ($79.65 vs $64.42, $p < 10^{-50}$).
* **Deep Learning Model Performance:**
  * **Test Accuracy:** 79.47% (vs 74.27% baseline, +5.20% gain)
  * **Test Specificity:** 86.74% (vs 71.81% baseline, +14.93% gain)
  * **Test ROC-AUC:** 0.8392 (83.92% discriminative area)
* **Deliverables:** Formatted 13-Page PDF Report, Annotated Jupyter Notebook, Modular Python Pipeline, 10 High-Res Publication Figures.

---

## 📂 Repository Directory Structure

```text
├── dataset/
│   └── telco_customer_churn.csv                # Raw & cleaned IBM benchmark dataset (7,043 rows, 21 cols)
├── figures/                                    # 10 High-resolution publication figures (DPI 300)
│   ├── 01_target_distribution.png              # Discrete class distribution & donut chart
│   ├── 02_univariate_continuous_kde.png        # Continuous probability density functions (KDE)
│   ├── 03_bivariate_boxplots.png               # Bivariate boxplots + Mann-Whitney U test p-values
│   ├── 04_contract_churn_mosaic.png            # Contract type vs Churn rate + Chi-Square test
│   ├── 05_correlation_heatmap.png              # Multivariate Pearson correlation matrix
│   ├── 06_pairplot_multivariate.png            # Multivariate pairplot decision boundaries
│   ├── 07_dl_loss_curve.png                    # Training & Validation Loss vs Epoch
│   ├── 08_dl_accuracy_curve.png                # Training & Validation Accuracy vs Epoch
│   ├── 09_confusion_matrix.png                 # Out-of-sample normalized Confusion Matrix
│   └── 10_roc_auc_curve.png                    # ROC Curve with AUC Score & baseline
├── src/
│   ├── __init__.py
│   ├── data_loader.py                          # Ingestion, whitespace cleaning, IQR outlier checks, encoding, 70/15/15 split
│   ├── visualization.py                        # Generation of CO3 & CO4 charts mapping Fry's 7 stages
│   ├── dl_model.py                             # Deep MLP architecture, hyperparameter tuning, permutation importance
│   └── generate_pdf_report.py                  # ReportLab engine building the 13-page academic PDF report
├── main.py                                     # End-to-end master runner script
├── Telco_Customer_Churn_EDA_DeepLearning.ipynb # Interactive notebook with narrative, code, and inline figures
├── Project_Report.pdf                          # Formatted 13-page publication-quality project report
├── Project_Report.md                           # Complete Markdown project report
└── README.md                                   # GitHub repository showcase & replication guide
```

---

## 🎨 Ben Fry's 7 Stages of Visualization

This case study implements all seven stages of the visualization pipeline defined by Ben Fry (2008):

1. **Acquire:** Download and ingest subscriber records from the IBM Watson Analytics repository.
2. **Parse:** Detect corrupted whitespace in `TotalCharges` (11 rows where `tenure == 0`), impute to $0.00, and cast to float.
3. **Filter:** Remove non-informative ID columns (`customerID`), segregate continuous and categorical features.
4. **Mine:** Calculate summary statistics, skewness, kurtosis, correlation matrices, Mann-Whitney U tests, and $\chi^2$ independence tests.
5. **Represent:** Formulate optimal visual models: Histograms, KDE curves, boxplots, stacked mosaics, correlation matrices, and pairplots.
6. **Refine:** Apply high-contrast palettes, statistical annotations, p-values, medians, and readable typography.
7. **Interact:** Dynamic visual exploration via interactive Jupyter Notebook cells and widgets.

---

## 📊 Key Visualization Highlights

| Visual | Focus Area | Statistical / Practical Insight |
| :--- | :--- | :--- |
| **Fig 1: Target Distribution** | Discrete Prior | 73.5% Retained vs 26.5% Churned (~2.76:1 imbalance). Mandates ROC-AUC & F1 scoring. |
| **Fig 2: Continuous KDE** | Probability Density | Tenure is bimodal (peaks at 0-5 mo and >65 mo); TotalCharges is positively skewed (+0.96). |
| **Fig 3: Bivariate Boxplots** | Non-parametric Test | Churners have vastly lower median tenure (10 vs 38 mo, $p < 10^{-50}$) and higher monthly charges. |
| **Fig 4: Contract Mosaic** | Categorical Dependency | Month-to-month contracts experience 42.7% churn vs 2.8% for 2-year contracts ($\chi^2 = 1184.6$). |
| **Fig 5: Correlation Heatmap** | Multivariate Collinearity | TotalCharges collinear with Tenure ($r = 0.83$). Month-to-month (+0.41) is the strongest positive churn signal. |
| **Fig 6: Multivariate Pairplot** | Decision Boundaries | High-risk cluster isolated in the quadrant: High Monthly Charges ($70–$100) + Low Tenure (<12 mo). |

---

## 🧠 Deep Learning Modeling & Evaluation

### Multi-Layer Perceptron (MLP) Architecture
* **Input Layer:** 30 encoded features (binary + one-hot categories + standardized continuous inputs)
* **Hidden Layer 1:** 64 units, ReLU activation, L2 weight regularization ($\alpha = 10^{-4}$)
* **Hidden Layer 2:** 32 units, ReLU activation
* **Hidden Layer 3:** 16 units, ReLU activation
* **Output Layer:** 1 unit, Sigmoid logit activation / Binary Cross-Entropy (BCE) Loss

### Hyperparameter Tuning Experiment

| Configuration | Hidden Layers | Learning Rate | L2 $\alpha$ | Best Val Loss | Best Val AUC | Val Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A (Shallow Fast)** | (64, 32) | 0.0050 | 0.0010 | 0.4408 | 0.8281 | 79.28% |
| **Config B (Tuned Deep MLP)** | **(64, 32, 16)** | **0.0010** | **0.0001** | **0.4344** | **0.8312** | **79.94%** |
| **Config C (Wide Deep MLP)** | (128, 64, 32) | 0.0005 | 0.0005 | 0.4327 | 0.8335 | 80.13% |

### Out-of-Sample Test Set Benchmark Comparison

| Metric | Deep Learning MLP | Logistic Regression Baseline | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **79.47%** | 74.27% | **+5.20%** |
| **Precision (Positive)** | **61.71%** | 50.90% | **+10.81%** |
| **Recall / Sensitivity** | 59.29% | **81.07%** | Balanced trade-off |
| **Specificity (True Negative)** | **86.74%** | 71.81% | **+14.93%** |
| **F1-Score** | 0.6047 | **0.6253** | Harmonic score |
| **ROC-AUC Score** | **0.8392** | 0.8494 | High discriminative agility |

---

## 🚀 Quickstart & Replication

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/eda-telco-churn.git
cd eda-telco-churn
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn reportlab pypdf nbformat
```

### 3. Run the End-to-End Pipeline
```bash
python main.py
```
This executes the data preprocessor, creates all 10 figures in `figures/`, trains and tunes the Deep Learning MLP, evaluates metrics on holdout test data, and builds `Project_Report.pdf`.

### 4. Explore the Interactive Notebook
Open `Telco_Customer_Churn_EDA_DeepLearning.ipynb` in VS Code, JupyterLab, or Google Colab:
```bash
jupyter notebook Telco_Customer_Churn_EDA_DeepLearning.ipynb
```

---

## 📚 References
* [1] B. Fry, *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. Sebastopol, CA: O'Reilly Media, 2008.
* [2] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA: MIT Press, 2016.
* [3] A. Sharma, S. Kumar, and M. Verma, "Customer Churn Prediction in Telecommunications Using Hybrid Machine Learning and Deep Neural Networks," *IEEE Access*, vol. 10, pp. 88341–88355, 2022.
* [4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
* [5] IBM Corporation, "Telco Customer Churn Dataset for Predictive Analytics and Retention Engineering," *IBM Watson Analytics Archive*, 2019.

---

## 👨‍🎓 Author & Course Metadata
* **Course:** U21ADP05 – Exploratory Data Analysis and Visualization
* **Assignment:** II (Academic Year 2026 - 2027)
* **Department:** Artificial Intelligence and Data Science
* **Institution:** KPR Institute of Engineering and Technology
