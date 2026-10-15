Exploratory Data Analysis, Statistical Visualization & Deep Learning on Telco Customer Churn

Python Framework Visuals Institution

A case study on customer attrition prediction applying Ben Fry's 7 Stages of Data Visualization, statistical hypothesis testing, and a Deep Learning Multi-Layer Perceptron (MLP) with hyperparameter tuning.

Submitted for U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II, 40 Marks), Department of Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology.

Student: Suja S (24AD120) | Date of submission: 15/10/2026

📌 Executive Summary
Domain & Application: Telecommunications subscriber retention and revenue optimization.
Dataset: IBM Telco Customer Churn (7,043 customer accounts, 21 features).
Key Findings:
Tenure and contract duration are the strongest drivers of retention.
Month-to-month contracts have a 42.7% churn rate, versus 11.3% for 1-year and 2.8% for 2-year contracts (χ² = 1184.6, p < 0.001).
Churners have higher median monthly charges ($79.65 vs $64.42; Mann-Whitney U, p < 10⁻⁵⁰).
Deep Learning Model (MLP) Test Performance:
Accuracy: 79.47% (logistic regression baseline: 74.27%, +5.20 percentage points)
Specificity: 86.74% (baseline: 71.81%, +14.93 percentage points)
ROC-AUC: 0.8392 (baseline: 0.8494)
Trade-off: the baseline has higher recall (81.07% vs 59.29%), so the better model depends on whether missed churners or wasted retention offers cost more.
Deliverables: 26-page PDF report (Telco_Churn_EDA_Case_Study_Report.pdf), annotated Jupyter notebook, modular Python pipeline (main.py, src/), standalone churn_pipeline.py, and 10 figures.
📂 Repository Structure
text
├── dataset/
│   └── telco_customer_churn.csv                # IBM Telco Customer Churn (7,043 rows, 21 columns)
├── figures/                                    # 10 figures (300 DPI)
│   ├── 01_target_distribution.png              # Discrete class distribution & donut chart
│   ├── 02_univariate_continuous_kde.png        # Continuous probability density functions (KDE)
│   ├── 03_bivariate_boxplots.png               # Boxplots + Mann-Whitney U test p-values
│   ├── 04_contract_churn_mosaic.png            # Contract type vs churn rate + Chi-Square test
│   ├── 05_correlation_heatmap.png              # Pearson correlation matrix
│   ├── 06_pairplot_multivariate.png            # Multivariate pairplot
│   ├── 07_dl_loss_curve.png                    # Training & validation loss vs epoch
│   ├── 08_dl_accuracy_curve.png                # Training & validation accuracy vs epoch
│   ├── 09_confusion_matrix.png                 # Test-set confusion matrix
│   └── 10_roc_auc_curve.png                    # ROC curves with AUC (MLP vs baseline)
├── src/
│   ├── data_loader.py                          # Ingestion, whitespace cleaning, IQR outlier checks, encoding, 70/15/15 split
│   ├── visualization.py                        # CO3 & CO4 charts mapped to Fry's 7 stages
│   ├── dl_model.py                             # MLP architecture, hyperparameter tuning, permutation importance
│   └── generate_pdf_report.py                  # ReportLab script that builds a PDF summary report
├── main.py                                     # End-to-end runner (pipeline used for the reported results)
├── churn_pipeline.py                           # Standalone, fully commented version of the whole pipeline
├── Telco_Customer_Churn_EDA_DeepLearning.ipynb # Notebook with narrative, code and inline figures
├── Telco_Churn_EDA_Case_Study_Report.pdf       # Final project report (26 pages)
└── README.md
🎨 Ben Fry's 7 Stages of Visualization

This case study follows the seven stages defined by Ben Fry (2008):

Acquire: Ingest subscriber records from the IBM sample dataset.
Parse: Detect blank strings in TotalCharges (11 rows, all with tenure == 0), impute $0.00, cast to float.
Filter: Drop the non-informative customerID; separate continuous and categorical features.
Mine: Summary statistics, skewness, kurtosis, correlation matrix, Mann-Whitney U tests and χ² independence tests.
Represent: Histograms and KDE curves, boxplots, stacked bars, a correlation heatmap and a pairplot.
Refine: Consistent palette, annotated p-values and medians, readable labels and titles.
Interact: Notebook-based exploration, plus an interactive Plotly scatter (figures/interactive_tenure_vs_charges.html) that churn_pipeline.py generates when Plotly is installed.
📊 Key Visualization Highlights
Visual	Focus Area	Statistical / Practical Insight
Fig 1: Target Distribution	Discrete prior	73.5% retained vs 26.5% churned (~2.76:1 imbalance), so ROC-AUC and F1 matter more than accuracy.
Fig 2: Continuous KDE	Probability density	Tenure is bimodal (peaks at 0–5 and >65 months); TotalCharges is positively skewed (+0.96).
Fig 3: Bivariate Boxplots	Non-parametric test	Churners have much lower median tenure (10 vs 38 months, p < 10⁻⁵⁰) and higher monthly charges.
Fig 4: Contract Mosaic	Categorical dependency	42.7% churn on month-to-month vs 2.8% on 2-year contracts (χ² = 1184.6).
Fig 5: Correlation Heatmap	Multicollinearity	TotalCharges is collinear with tenure (r = 0.83); month-to-month (+0.41) is the strongest positive churn correlate.
Fig 6: Pairplot	Multivariate structure	Churners cluster at high monthly charges ($70–$100) combined with low tenure (<12 months).
🧠 Deep Learning Modeling & Evaluation
Multi-Layer Perceptron (MLP) Architecture
Input: 30 encoded features (binary + one-hot categories + standardized continuous inputs)
Hidden layer 1: 64 units, ReLU, L2 weight regularization (α = 10⁻⁴)
Hidden layer 2: 32 units, ReLU
Hidden layer 3: 16 units, ReLU
Output: 1 unit, sigmoid / binary cross-entropy loss
Parameters: 4,609 (Config B)
Hyperparameter Tuning Experiment
Configuration	Hidden Layers	Learning Rate	L2 α	Best Val Loss	Best Val AUC	Val Accuracy
Config A (Shallow Fast)	(64, 32)	0.0050	0.0010	0.4408	0.8281	79.28%
Config B (Tuned Deep MLP)	(64, 32, 16)	0.0010	0.0001	0.4344	0.8312	79.94%
Config C (Wide Deep MLP)	(128, 64, 32)	0.0005	0.0005	0.4327	0.8335	80.13%
Out-of-Sample Test Set Comparison (1,057 customers)
Metric	Deep Learning MLP	Logistic Regression Baseline	Difference
Accuracy	79.47%	74.27%	+5.20 points
Precision	61.71%	50.90%	+10.81 points
Recall / Sensitivity	59.29%	81.07%	−21.78 points
Specificity	86.74%	71.81%	+14.93 points
F1-Score	0.6047	0.6253	−0.0206
ROC-AUC	0.8392	0.8494	−0.0102

The MLP is more precise and avoids more false alarms; the baseline catches more churners. The two are close in ROC-AUC.

Reproducibility Note

Re-running churn_pipeline.py reproduces the data-quality results, the split counts (train 4,929 / val 1,057 / test 1,057), and the logistic-regression baseline exactly. MLP test metrics from that re-run (accuracy 79.66%, specificity 87.77%, ROC-AUC 0.8474) differ from the reported ones by about one point, which is expected for neural networks with different random initialization or library versions.

🚀 Quickstart & Replication
1. Clone the Repository
bash
git clone https://github.com/sujatamil2006-star/EDA.git
cd EDA
2. Install Dependencies
bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn plotly reportlab pypdf nbformat
3. Run the Pipeline
bash
python main.py              # pipeline used for the reported results
python churn_pipeline.py    # standalone commented version (prints all tables, saves figures/)

Both read dataset/telco_customer_churn.csv, create the figures in figures/, train and tune the MLP, and evaluate it on the held-out test set. Note that running either script overwrites the files in figures/.

4. Explore the Notebook
bash
jupyter notebook Telco_Customer_Churn_EDA_DeepLearning.ipynb
📁 Dataset
Name: Telco Customer Churn (IBM sample dataset)
Source (IBM GitHub): https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
Direct download: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Local copy: dataset/telco_customer_churn.csv
Size: 7,043 customers, 21 features (target: Churn)
📚 References
[1] B. Fry, Visualizing Data: Exploring and Explaining Data with the Processing Environment. Sebastopol, CA: O'Reilly Media, 2008.
[2] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA: MIT Press, 2016.
[3] A. K. Ahmad, A. Jafar, and K. Aljoumaa, "Customer churn prediction in telecom using machine learning in big data platform," Journal of Big Data, vol. 6, art. no. 28, pp. 1–24, 2019, doi: 10.1186/s40537-019-0191-6.
[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
[5] IBM, "Predict Customer Churn using Watson Machine Learning and Jupyter Notebooks on Cloud Pak for Data (Telco Customer Churn dataset)," GitHub repository. [Online]. Available: https://github.com/IBM/telco-customer-churn-on-icp4d
👩‍🎓 Author & Course Metadata
Student: Suja S (24AD120)
Course: U21ADP05 – Exploratory Data Analysis and Visualization
Assignment: II (Academic Year 2026 – 2027, Semester V, Batch 2023 – 2027)
Course Coordinator: Mr. Rushikesh Kadam
Department: Artificial Intelligence and Data Science
Institution: KPR Institute of Engineering and Technology