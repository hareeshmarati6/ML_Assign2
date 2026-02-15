# app.py
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Breast Cancer Diagnosis — Classifiers", layout="wide")

MODELS_DIR = Path("models")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
}

st.title("🔬 Breast Cancer (Diagnostic) — Classification")
st.caption("Upload test CSV (same schema as training, with 'target' column), choose a model, and view metrics + confusion matrix.")

# Sidebar
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def multiclass_auc(y_true, proba, classes):
    lb = LabelBinarizer()
    lb.fit(classes)
    y_bin = lb.transform(y_true)
    try:
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        import numpy as np
        auc = np.nan
    return auc

def plot_cm(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=sorted(np.unique(y_true)),
                yticklabels=sorted(np.unique(y_true)), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

st.header("1) Upload Test CSV")
uploaded = st.file_uploader("Upload a CSV with the same columns used in training (features + 'target' as label).", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, sep=None, engine="python")  # auto-detect delimiter
    st.write("Preview:", df.head())

    # Basic schema checks
    if "target" not in df.columns:
        st.error("CSV must contain the target column 'target'.")
        st.stop()

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    classes_sorted = sorted(y.unique())

    model_path = MODELS_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}. Please run models.py first and commit models/ to repo.")
        st.stop()

    model = load_model(model_path)

    # Predict
    y_pred = model.predict(X)
    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        # fallback: one-hot of predictions
        import numpy as np
        proba = np.zeros((len(y_pred), len(classes_sorted)))
        class_to_index = {c:i for i,c in enumerate(classes_sorted)}
        for i, p in enumerate(y_pred):
            proba[i, class_to_index.get(p, 0)] = 1.0

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    auc = multiclass_auc(y, proba, classes_sorted)

    st.header("2) Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("Precision (macro)", f"{prec:.4f}")
    with c2:
        st.metric("Recall (macro)", f"{rec:.4f}")
        st.metric("F1 (macro)", f"{f1:.4f}")
    with c3:
        st.metric("AUC (macro OVR)", f"{auc:.4f}" if not pd.isna(auc) else "NA")
        st.metric("MCC", f"{mcc:.4f}")

    st.header("3) Confusion Matrix")
    plot_cm(y, y_pred, title=f"Confusion Matrix — {model_name}")

    st.header("4) Classification Report")
    st.text(classification_report(y, y_pred, digits=4))
else:
    st.info("Upload a CSV with the same schema as training to evaluate.\nTip: After running `models.py`, you can upload the generated `models/test_results.csv` here.")
