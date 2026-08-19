# Project Context — AI Mental Health Support Chatbot

This document serves as the authoritative state of the **AI Mental Health Support Chatbot** project across development environments and coding platforms.

---

## 1. Project Objective
Build an empathetic AI Mental Health Support Chatbot that categorizes user statements into mental health categories with high confidence and provides compassionate, non-diagnostic conversational responses.

---

## 2. Locked Architecture
The application architecture is strictly locked as follows:

```
User Statement
  │
  ▼
Streamlit UI
  │
  ▼
Fine-tuned DistilBERT Classifier  ───► Predicted mental health category + confidence score
  │
  ▼
Gemini API (Conversational Model) ───► Compassionate response contextualized by category
  │
  ▼
Streamlit UI
```

### Constraints & Architectural Boundaries:
- **DistilBERT**: Used ONLY as the text classification model.
- **Gemini API**: Used ONLY as the conversational response generation model.
- **No RAG**: Retrieval-Augmented Generation is NOT permitted.
- **No Embeddings / Vector Databases**: Vector search or custom embedding index is NOT permitted.
- **No Gemini Classification**: Gemini must not be used as the classifier.
- **No Model Swapping**: DistilBERT must not be replaced with another final classifier.

---

## 3. Current Phase
**PHASE 2 — BASELINE TEXT CLASSIFIER** (Completed)

---

## 4. Dataset Source & Status
- **Kaggle Dataset**: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- **Raw Path**: `data/raw/Combined Data.csv` (`53,043` rows)
- **Cleaned Path**: `data/processed/cleaned_mental_health_data.csv` (`51,055` rows)

---

## 5. Environment & Setup Details
- **Python Version**: `3.13.3`
- **Virtual Environment**: `.venv`
- **Installed Package Versions**:
  - `pandas`: `3.0.5`
  - `numpy`: `2.5.2`
  - `scikit-learn`: `1.9.0`
  - `matplotlib`: `3.11.1`
  - `seaborn`: `0.13.2`
  - `joblib`: `1.5.3`
  - `jupyter`: `1.1.1` (notebook `7.6.2`)

---

## 6. Project Structure & Files Created
```
mental-health-chatbot/
│
├── .venv/                         # Local virtual environment (ignored by git)
│
├── data/
│   ├── raw/
│   │   └── Combined Data.csv     # Raw Kaggle dataset (53,043 rows)
│   └── processed/
│       ├── cleaned_mental_health_data.csv # Processed dataset (51,055 rows)
│       └── splits/               # Persisted train/val/test CSV splits
│           ├── train.csv         # 40,844 rows (80%)
│           ├── validation.csv    # 5,105 rows (10%)
│           └── test.csv          # 5,106 rows (10%)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb # Data exploration & cleaning documentation
│   └── 02_baseline_classifier.ipynb # Baseline classifier experiments & analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cleaning.py          # Modular, executable dataset cleaning pipeline
│   └── models/
│       ├── __init__.py
│       └── baseline.py          # Modular baseline training & evaluation pipeline
│
├── models/
│   └── baseline/                # Persisted vectorizer, metadata & model artifacts
│       ├── metadata.json         # Complete experiment reproducibility parameters
│       ├── tfidf_vectorizer.joblib # Fitted TF-IDF vectorizer (274,360 features)
│       ├── logistic_regression.joblib # Trained Logistic Regression model
│       ├── linear_svc.joblib     # Trained LinearSVC model (Selected Baseline)
│       ├── cm_val_logistic_regression.png # Validation Confusion Matrix
│       ├── cm_val_linear_svc.png  # Validation Confusion Matrix
│       └── cm_test_selected_baseline.png # Test Confusion Matrix
│
├── app/
│   └── .gitkeep                  # Placeholder for future Streamlit UI app
│
├── PROJECT_CONTEXT.md            # Authoritative project state document
├── README.md                     # Setup and execution guide
├── requirements.txt              # Phase 1 & 2 dependencies
└── .gitignore                    # Version control exclusion rules
```

---

## 7. Data Splits & Label Consistency

### Stratified Splits (80% / 10% / 10%, `random_state=42`)
- **Train Set**: `40,844` rows (`80.00%`)
- **Validation Set**: `5,105` rows (`10.00%`)
- **Test Set**: `5,106` rows (`10.00%`)
- **Total Rows**: `51,055` rows
- **Data Leakage Check**: Confirmed **zero statement overlap** across train, validation, and test splits.

### Canonical Label Set & Ordering (7 Classes)
Fixed canonical order used consistently across all evaluation reports, confusion matrices, and future phases:
1. `Anxiety`
2. `Bipolar`
3. `Depression`
4. `Normal`
5. `Personality disorder`
6. `Stress`
7. `Suicidal`

### Class Distribution Across Splits

| Target Label | Total Count | Train Count (80%) | Validation Count (10%) | Test Count (10%) |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 16,037 | 12,830 | 1,603 | 1,604 |
| **Depression** | 15,078 | 12,062 | 1,508 | 1,508 |
| **Suicidal** | 10,634 | 8,507 | 1,063 | 1,064 |
| **Anxiety** | 3,617 | 2,894 | 361 | 362 |
| **Bipolar** | 2,501 | 2,001 | 250 | 250 |
| **Stress** | 2,293 | 1,834 | 230 | 229 |
| **Personality disorder** | 895 | 716 | 90 | 89 |
| **TOTAL** | **51,055** | **40,844** | **5,105** | **5,106** |

---

## 8. Feature Extraction & Baseline Models Setup

### TF-IDF Vectorizer Configuration
- `ngram_range`: `(1, 2)` (unigrams + bigrams)
- `min_df`: `2`, `max_df`: `0.95`
- `sublinear_tf`: `True`
- **Fitted ONLY on Train set** text: Produced `274,360` vocabulary features.

### Classifier Models
1. **Logistic Regression**: `class_weight='balanced'`, `random_state=42`, `max_iter=1000`
2. **LinearSVC**: `class_weight='balanced'`, `random_state=42`, `max_iter=2000`

---

## 9. Validation Results & Baseline Selection

### Validation Metrics Comparison (Primary Metric: Macro F1)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7798 | 0.7218 | **0.7570** | 0.7349 | **0.7873** | 0.7798 | 0.7796 |
| **LinearSVC (WINNER)** | **0.7843** | **0.7697** | 0.7201 | **0.7392** | 0.7825 | **0.7843** | **0.7821** |

### Winning Baseline Selection:
- **Selected Winner**: **LinearSVC** (Validation Macro F1 = `0.7392` vs Logistic Regression = `0.7349`).

---

## 10. Per-Class Performance Breakdown

### Validation Set Per-Class Results (LinearSVC):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.8974 | 0.9495 | **0.9227** | 1,603 |
| **Anxiety** | 0.7932 | 0.8393 | **0.8156** | 361 |
| **Bipolar** | 0.8578 | 0.7720 | **0.8126** | 250 |
| **Depression** | 0.7424 | 0.6976 | **0.7193** | 1,508 |
| **Suicidal** | 0.6828 | 0.7168 | **0.6994** | 1,063 |
| **Personality disorder** | 0.8246 | 0.5222 | **0.6395** | 90 |
| **Stress** | 0.5896 | 0.5435 | **0.5656** | 230 |

### Final Test Set Evaluation (LinearSVC — Evaluated EXACTLY ONCE):
- **Test Accuracy**: `0.7763`
- **Test Macro Precision**: `0.7522`
- **Test Macro Recall**: `0.6959`
- **Test Macro F1**: `0.7165`
- **Test Weighted F1**: `0.7737`

#### Test Set Per-Class Detailed Results:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.8986 | 0.9451 | **0.9213** | 1,604 |
| **Anxiety** | 0.7842 | 0.8232 | **0.8032** | 362 |
| **Bipolar** | 0.7675 | 0.7000 | **0.7322** | 250 |
| **Depression** | 0.7302 | 0.7036 | **0.7166** | 1,508 |
| **Suicidal** | 0.6730 | 0.7002 | **0.6863** | 1,064 |
| **Stress** | 0.6517 | 0.5721 | **0.6093** | 229 |
| **Personality disorder** | 0.7600 | 0.4270 | **0.5468** | 89 |

---

## 11. Key Observations & Known Baseline Limitations
1. **Majority vs Minority Class Gap**: `Normal` achieves an F1 of `0.9213` on the test set, whereas minority classes like `Personality disorder` (F1 `0.5468`) and `Stress` (F1 `0.6093`) suffer from limited training instances despite class-weight balancing.
2. **Semantic Ambiguity**: Overlap between `Depression` and `Suicidal` statements leads to confusion in linear bag-of-words models due to shared vocabulary ("hopeless", "sad", "give up").
3. **Motivation for Phase 3 (DistilBERT)**: Deep contextual embeddings from DistilBERT are expected to improve contextual distinction on overlapping and minority mental health categories.

---

## 12. Exact Commands to Reproduce Phase 2

### Virtual Environment Activation:
```bash
# Windows (PowerShell):
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

### Run Baseline Pipeline:
```bash
python -m src.models.baseline
```

### Run Baseline Notebook:
```bash
jupyter notebook notebooks/02_baseline_classifier.ipynb
```

---

## 13. Next Phase
**PHASE 3 — FINE-TUNE DISTILBERT CLASSIFIER**
- Fine-tune `distilbert-base-uncased` on `data/processed/splits/train.csv`.
- Evaluate on `data/processed/splits/validation.csv`.
- Compare DistilBERT performance against Phase 2 LinearSVC baseline (`Macro F1 = 0.7392`).
