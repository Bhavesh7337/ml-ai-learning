import numpy as np
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


DATA_DIR = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data"
REPORTS_DIR = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\reports"
MODELS_DIR = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\models"


def ensure_dirs():
    """Ensure required directories exist."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)



def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n {model_name} Performance:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"{model_name.lower()}_confusion_matrix.png"))
    plt.close()

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc
    }


# ==========================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    ensure_dirs()

    model_defs = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    all_metrics = []

    for name, model in model_defs.items():
        print(f"\n Training and evaluating {name}...")
        model.fit(X_train, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

        # Save trained model
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"✅ {name} model saved to {model_path}")

    
    metrics_df = pd.DataFrame(all_metrics)
    print("\n All Model Metrics:")
    print(metrics_df)

    # Save metrics to CSV and JSON
    metrics_df.to_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"), index=False)
    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    best_model = max(all_metrics, key=lambda x: x["accuracy"])
    print(f"\n Best Model: {best_model['model']} with Accuracy: {best_model['accuracy']:.4f}")

    return best_model, all_metrics



if __name__ == "__main__":
    print("Loading preprocessed data...")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print("Data loaded.")
    train_and_evaluate(X_train, X_test, y_train, y_test)
