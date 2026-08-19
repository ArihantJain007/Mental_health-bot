import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Canonical label ordering fixed across Phase 2 and future phases
CANONICAL_LABELS = [
    'Anxiety',
    'Bipolar',
    'Depression',
    'Normal',
    'Personality disorder',
    'Stress',
    'Suicidal'
]


def get_canonical_labels() -> list:
    """Returns fixed canonical list of target mental health categories."""
    return list(CANONICAL_LABELS)


def create_stratified_splits(df_path: str = "data/processed/cleaned_mental_health_data.csv",
                            output_dir: str = "data/processed/splits",
                            random_state: int = 42) -> tuple:
    """Creates reproducible 80% train, 10% validation, 10% test stratified splits."""
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {df_path}")
        
    df = pd.read_csv(df_path)
    
    # 1. First split: 80% train, 20% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        stratify=df['status']
    )
    
    # 2. Second split: 50% val, 50% test of temp_df -> 10% val, 10% test total
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df['status']
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Verification assertions
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Split size sum mismatch!"
    
    train_set = set(train_df['statement'])
    val_set = set(val_df['statement'])
    test_set = set(test_df['statement'])
    
    assert len(train_set.intersection(val_set)) == 0, "Data leakage! Train and Val overlap."
    assert len(train_set.intersection(test_set)) == 0, "Data leakage! Train and Test overlap."
    assert len(val_set.intersection(test_set)) == 0, "Data leakage! Val and Test overlap."
    
    return train_df, val_df, test_df


def save_reproducibility_metadata(filepath: str = "models/baseline/metadata.json",
                                 cleaned_row_count: int = 51055,
                                 random_state: int = 42) -> dict:
    """Saves complete experiment reproducibility metadata to JSON."""
    metadata = {
        "random_state": random_state,
        "train_fraction": 0.80,
        "validation_fraction": 0.10,
        "test_fraction": 0.10,
        "tfidf_config": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True
        },
        "canonical_label_ordering": get_canonical_labels(),
        "dataset_source_path": "data/processed/cleaned_mental_health_data.csv",
        "cleaned_dataset_row_count": cleaned_row_count
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata


def fit_tfidf_vectorizer(train_texts: pd.Series, config: dict = None) -> TfidfVectorizer:
    """Fits TfidfVectorizer ONLY on training text."""
    if config is None:
        config = {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True
        }
    vectorizer = TfidfVectorizer(**config)
    vectorizer.fit(train_texts)
    return vectorizer


def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Trains multiclass LogisticRegression with class_weight='balanced'."""
    model = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=1000,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svc(X_train, y_train, random_state: int = 42) -> LinearSVC:
    """Trains LinearSVC with class_weight='balanced'."""
    model = LinearSVC(
        class_weight="balanced",
        random_state=random_state,
        max_iter=2000
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_eval, y_eval, labels_order: list = None) -> dict:
    """Evaluates classifier model and returns comprehensive metric dictionary."""
    if labels_order is None:
        labels_order = get_canonical_labels()
        
    y_pred = model.predict(X_eval)
    
    acc = accuracy_score(y_eval, y_pred)
    
    # Macro metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, labels=labels_order, average="macro", zero_division=0
    )
    
    # Weighted metrics
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, labels=labels_order, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_eval, y_pred, labels=labels_order, average=None, zero_division=0
    )
    
    per_class = {}
    for idx, label in enumerate(labels_order):
        per_class[label] = {
            "precision": float(per_p[idx]),
            "recall": float(per_r[idx]),
            "f1": float(per_f1[idx]),
            "support": int(per_sup[idx])
        }
        
    cm = confusion_matrix(y_eval, y_pred, labels=labels_order)
    
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred.tolist()
    }


def plot_and_save_confusion_matrix(cm_list: list, labels: list, title: str, output_path: str):
    """Plots and saves confusion matrix heatmap."""
    cm = np.array(cm_list)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_model_artifact(artifact, filepath: str):
    """Saves model artifact using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(artifact, filepath)


def load_model_artifact(filepath: str):
    """Loads model artifact using joblib."""
    return joblib.load(filepath)


def run_baseline_pipeline() -> dict:
    """Executes the complete Phase 2 baseline modeling pipeline."""
    print("=" * 70)
    print("PHASE 2 — BASELINE TEXT CLASSIFIER PIPELINE")
    print("=" * 70)
    
    # 1. Create Stratified Splits
    print("\n1. Generating Stratified Data Splits (80/10/10)...")
    train_df, val_df, test_df = create_stratified_splits()
    print(f"   - Train set shape: {train_df.shape}")
    print(f"   - Validation set shape: {val_df.shape}")
    print(f"   - Test set shape: {test_df.shape}")
    
    labels = get_canonical_labels()
    print(f"\n2. Canonical Labels ({len(labels)} classes):")
    print("   ", labels)
    
    # Save reproducibility metadata
    save_reproducibility_metadata(
        filepath="models/baseline/metadata.json",
        cleaned_row_count=len(train_df) + len(val_df) + len(test_df)
    )
    print("   - Saved metadata to models/baseline/metadata.json")
    
    # 2. Fit TF-IDF Vectorizer
    print("\n3. Fitting TF-IDF Vectorizer on Training Data ONLY...")
    vectorizer = fit_tfidf_vectorizer(train_df['statement'])
    X_train = vectorizer.transform(train_df['statement'])
    X_val = vectorizer.transform(val_df['statement'])
    X_test = vectorizer.transform(test_df['statement'])
    
    print(f"   - TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_)} features")
    save_model_artifact(vectorizer, "models/baseline/tfidf_vectorizer.joblib")
    print("   - Saved vectorizer to models/baseline/tfidf_vectorizer.joblib")
    
    # 3. Train & Evaluate Logistic Regression
    print("\n4. Training Logistic Regression (class_weight='balanced')...")
    lr_model = train_logistic_regression(X_train, train_df['status'])
    save_model_artifact(lr_model, "models/baseline/logistic_regression.joblib")
    print("   - Saved model to models/baseline/logistic_regression.joblib")
    
    lr_val_eval = evaluate_model(lr_model, X_val, val_df['status'], labels)
    print(f"   - Validation Accuracy:    {lr_val_eval['accuracy']:.4f}")
    print(f"   - Validation Macro F1:    {lr_val_eval['macro_f1']:.4f}")
    print(f"   - Validation Weighted F1: {lr_val_eval['weighted_f1']:.4f}")
    
    # 4. Train & Evaluate LinearSVC
    print("\n5. Training LinearSVC (class_weight='balanced')...")
    svc_model = train_linear_svc(X_train, train_df['status'])
    save_model_artifact(svc_model, "models/baseline/linear_svc.joblib")
    print("   - Saved model to models/baseline/linear_svc.joblib")
    
    svc_val_eval = evaluate_model(svc_model, X_val, val_df['status'], labels)
    print(f"   - Validation Accuracy:    {svc_val_eval['accuracy']:.4f}")
    print(f"   - Validation Macro F1:    {svc_val_eval['macro_f1']:.4f}")
    print(f"   - Validation Weighted F1: {svc_val_eval['weighted_f1']:.4f}")
    
    # 5. Model Selection based on Validation Macro F1
    print("\n6. Model Comparison & Selection (Validation Macro F1):")
    print(f"   - Logistic Regression Macro F1: {lr_val_eval['macro_f1']:.4f}")
    print(f"   - LinearSVC Macro F1:           {svc_val_eval['macro_f1']:.4f}")
    
    if lr_val_eval['macro_f1'] >= svc_val_eval['macro_f1']:
        winning_name = "Logistic Regression"
        winning_model = lr_model
        winning_val_eval = lr_val_eval
    else:
        winning_name = "LinearSVC"
        winning_model = svc_model
        winning_val_eval = svc_val_eval
        
    print(f"\n   ===> SELECTED WINNING BASELINE: {winning_name} <===")
    
    # 6. Evaluate Selected Model ONCE on Test Set
    print(f"\n7. Evaluating Selected Model ({winning_name}) ONCE on Test Set...")
    test_eval = evaluate_model(winning_model, X_test, test_df['status'], labels)
    print(f"   - Test Accuracy:    {test_eval['accuracy']:.4f}")
    print(f"   - Test Macro F1:    {test_eval['macro_f1']:.4f}")
    print(f"   - Test Weighted F1: {test_eval['weighted_f1']:.4f}")
    
    # Plot and save confusion matrices
    plot_and_save_confusion_matrix(
        lr_val_eval['confusion_matrix'], labels,
        "Validation Confusion Matrix - Logistic Regression",
        "models/baseline/cm_val_logistic_regression.png"
    )
    plot_and_save_confusion_matrix(
        svc_val_eval['confusion_matrix'], labels,
        "Validation Confusion Matrix - LinearSVC",
        "models/baseline/cm_val_linear_svc.png"
    )
    plot_and_save_confusion_matrix(
        test_eval['confusion_matrix'], labels,
        f"Test Confusion Matrix - Selected Baseline ({winning_name})",
        "models/baseline/cm_test_selected_baseline.png"
    )
    
    print("\nAll baseline artifacts and confusion matrix plots saved successfully!")
    print("=" * 70)
    
    return {
        "lr_val": lr_val_eval,
        "svc_val": svc_val_eval,
        "winning_name": winning_name,
        "test_eval": test_eval
    }


if __name__ == "__main__":
    run_baseline_pipeline()
