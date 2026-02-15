# Breast Cancer Diagnosis — Classification

This repository trains multiple classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset, exports artifacts, and exposes a Streamlit UI (`app.py`) to evaluate models on a test CSV.

---

## Problem statement

Predict breast cancer diagnosis (malignant vs benign) from features computed on digitized FNA (fine needle aspirate) images. This is a binary classification task: `0 = malignant`, `1 = benign`.

## Dataset description [1 mark]

- **Dataset name:** Breast Cancer Wisconsin (Diagnostic)
- **Source / loader:** `sklearn.datasets.load_breast_cancer` (cached to `data/breast_cancer_wisconsin.csv` when `train_models.py` is run)
- **Instances:** 569
- **Features:** 30 numeric input features (plus `target` label)
- **Target mapping:** `0 = malignant`, `1 = benign`

## Models used & Evaluation

The project trains the following models and computes the listed metrics on a common test split (saved to `models/metrics_summary.csv`): Accuracy, AUC (macro/OVR where applicable), Precision (macro), Recall (macro), F1 (macro), and Matthews Correlation Coefficient (MCC).

Trained models:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table (fill after training) [6 marks]

Copy metric values from `models/metrics_summary.csv` into the table below.

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.986013986		0.985010482	0.985010482	0.985010482	0.970020964
XGBoost (Ensemble)	0.965034965		0.968494624	0.956708595	0.962066953	0.925128146
Random Forest (Ensemble)	0.958041958		0.958653026	0.95115304	0.95467033	0.909775153
KNN	0.958041958		0.96875	0.943396226	0.953870968	0.911793795
Naive Bayes	0.937062937		0.942324561	0.922851153	0.93112122	0.864956533
Decision Tree	0.923076923		0.914511494	0.923375262	0.918467838	0.837839871


## Observations (3 marks)

Provide concise observations for each model based on the metrics above. Examples are placeholders — replace with your actual findings.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | May underperform on strong non-linear patterns; benefits from scaling. |
| Decision Tree | Can overfit without pruning; interpretable splits. |
| kNN | Sensitive to scaling and `k`; performance varies with `k`. |
| Naive Bayes | Very fast; independence assumption can limit performance on correlated features. |
| Random Forest (Ensemble) | Robust baseline; typically high F1 and stable across splits. |
| XGBoost (Ensemble) | Often top performer after tuning; handles complex interactions well. |

## How to run (quick)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models and generate artifacts:

```bash
python train_models.py
```

Artifacts are saved under `models/`:
- `*.joblib` — trained models
- `metrics_summary.csv` — aggregated metrics for the table above
- `test_holdout.csv` — test CSV for upload to the app

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` and upload `models/test_holdout.csv` to reproduce evaluation.

## Submission note

Include the contents of this README (Problem statement, Dataset description, comparison table, and observations) in your submitted PDF file.

## Repo structure (summary)

```
project-folder/
├─ app.py
├─ train_models.py
├─ requirements.txt
├─ README.md
├─ models/
│  ├─ *.joblib
│  ├─ metrics_summary.csv
│  ├─ test_holdout.csv
│  └─ figs/*_cm.png
└─ data/
   └─ breast_cancer_wisconsin.csv
```

## References

- sklearn `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
# Breast Cancer Diagnosis Classification (UCI / sklearn)

This project implements and deploys six classification models on the **Breast Cancer Wisconsin (Diagnostic)** dataset. It follows the assignment requirements: modeling, evaluation, Streamlit UI, and deployment.

## Problem Statement
Predict **breast cancer diagnosis** (malignant or benign) using computed features from digitized images of fine needle aspirate (FNA) of breast masses. This is framed as a **binary classification** problem. (Dataset source: UCI / sklearn)

**Dataset (sklearn loader):** `sklearn.datasets.load_breast_cancer`

The dataset contains **569 instances** and **30 numeric input features** (plus the `target` label: 0=malignant, 1=benign).

## Dataset Description ✅
- **Source:** UCI Machine Learning Repository — Wine Quality (White)  
- **Download (CSV):** `winequality-white.csv` (semicolon `;` separated)  
- **Instances:** 4,898  
- **Features:** 12 (11 inputs: fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol; **target**: quality [0–10])  
- **Task:** Multiclass classification (quality levels)  
- **Note:** We treat the problem as **multiclass classification** (not regression). Class distribution is skewed; macro-averaged metrics are reported.

## Models Used & Metrics (to be filled after training)
We trained the following models on the same data split:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

**Reported metrics on the test split:** Accuracy, **AUC (macro OVR)**, Precision (macro), Recall (macro), F1 (macro), **MCC**.

After you run `python train_models.py`, copy values from `models/metrics_summary.csv` into the table below:

| ML Model Name | Accuracy | AUC (macro) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| XGBoost (Ensemble) | | | | | | |

## Observations (add after training)
Typical observations for this dataset:

- **Logistic Regression** – Benefits from scaling; can underfit non-linearities but gives stable macro metrics.  
- **Decision Tree** – Captures non-linear boundaries; may overfit without constraints.  
- **kNN** – Sensitive to scaling and `k`; mid-range neighbors often perform well.  
- **Naive Bayes** – Very fast; independence assumption can hurt with correlated features (e.g., SO2 measures).  
- **Random Forest** – Strong tabular baseline; robust and often among top F1.  
- **XGBoost** – Typically best or close; tune learning rate, depth, estimators for gains.

## How to Run (BITS Virtual Lab)

```bash
# 1) Install dependencies
pip install -r requirements.txt

2) (Optional) Cache the CSV locally
mkdir data
# (Dataset is already included as breast_cancer_wisconsin.csv in the repo)

3) Train all models, compute metrics, export artifacts
python models.py

# Artifacts saved to ./models
# - *.joblib (six models)

This project trains and exposes several classification models for the **Breast Cancer Wisconsin (Diagnostic)** dataset. The code implements training, evaluation, artifact export, and a Streamlit UI to upload test CSVs and view model performance.

## Problem Statement
Predict **breast cancer diagnosis** (malignant or benign) using features computed from digitized images of fine needle aspirate (FNA) of breast masses. This is a **binary classification** task using the dataset loaded via `sklearn.datasets.load_breast_cancer`.

## Dataset
- **Loader:** `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Features:** 30 numeric input features (plus `target` label)
- **Target mapping:** `0 = malignant`, `1 = benign`

The repository caches the dataset to `data/breast_cancer_wisconsin.csv` when `models.py` is run.

## Models Trained
The project trains the following classifiers (saved to `models/`):

1. Logistic Regression (with `StandardScaler`)
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN, with `StandardScaler`)
4. Gaussian Naive Bayes
5. Random Forest (ensemble)
6. XGBoost (ensemble)

## Metrics Computed
For each model the script computes and saves:
- Accuracy
- AUC (macro, one-vs-rest)
- Precision (macro)
- Recall (macro)
- F1 (macro)
- Matthews Correlation Coefficient (MCC)

Per-model classification reports are written to `models/*_report.txt`; confusion matrices are saved under `models/figs/`; an aggregated `models/metrics_summary.csv` is produced.

## How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models and generate artifacts:

```bash
python models.py
```

Artifacts created under the `models/` directory:

- `*.joblib` — saved model objects
- `metrics_summary.csv` — aggregated metrics table
- `test_holdout.csv` — test CSV ready for upload to the app
- `figs/` — confusion matrix images

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Open the app, choose a model from the sidebar, and upload the `models/test_holdout.csv` file to reproduce evaluation.

## Streamlit App Features
- CSV upload (test CSV with the same schema including `target`)
- Model selection dropdown
- Display of evaluation metrics, confusion matrix, and classification report

## Deployment
If deploying to Streamlit Community Cloud or similar:
1. Ensure `requirements.txt` includes required packages (`xgboost` is required for XGBoost model).
2. Commit the trained `models/*.joblib` files to the repository, or modify the app to download models from remote storage at startup.

## Repo Structure

```
project-folder/
├─ streamlit_app.py
├─ models.py
├─ requirements.txt
├─ README.md
├─ breast_cancer_wisconsin.csv
├─ models/
│  ├─ *.joblib
│  ├─ metrics_summary.csv
│  ├─ test_holdout.csv
│  ├─ *.txt
│  └─ figs/
└─ data/
   └─ breast_cancer_wisconsin.csv
```

## Notes
- The codebase uses the Breast Cancer dataset (binary classification). If you intended to work on the Wine Quality dataset instead, I can update the scripts to load and treat Wine Quality as a multiclass task.

## References
- sklearn `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
