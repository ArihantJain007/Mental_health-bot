# AI Mental Health Support Chatbot

An empathetic, AI-driven mental health support application that classifies user text into mental health categories with high confidence and generates supportive, non-diagnostic conversational responses.

---

## 🔒 Locked Architecture

```
User Input ──► Streamlit UI ──► Fine-tuned DistilBERT ──► Predicted Category + Confidence ──► Gemini API ──► Empathetic Response ──► Streamlit UI
```

- **Classifier Model**: Fine-tuned DistilBERT (category prediction & confidence scoring only).
- **Conversational Model**: Gemini API (empathetic response generation only).
- **Restrictions**: No RAG, no vector databases, no embeddings, no Gemini classification, no model swapping.

---

## 📌 Project Status & Phases

- **Phase 1: Dataset Exploration & Cleaning** (Completed)
- **Phase 2: Baseline Text Classifier** (Completed)
- **Phase 3: Fine-tune DistilBERT Classifier** (Next)

---

## 📁 Directory Structure

```
mental-health-chatbot/
│
├── .venv/                         # Python virtual environment (ignored by git)
│
├── data/
│   ├── raw/
│   │   └── Combined Data.csv     # Raw untouched dataset (53,043 rows)
│   └── processed/
│       ├── cleaned_mental_health_data.csv # Processed dataset (51,055 rows)
│       └── splits/               # Stratified 80/10/10 data splits
│           ├── train.csv         # 40,844 rows
│           ├── validation.csv    # 5,105 rows
│           └── test.csv          # 5,106 rows
│
├── notebooks/
│   ├── 01_data_exploration.ipynb # Data exploration & cleaning notebook
│   └── 02_baseline_classifier.ipynb # Baseline classifier notebook
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cleaning.py          # Modular dataset cleaning pipeline
│   └── models/
│       ├── __init__.py
│       └── baseline.py          # Baseline modeling pipeline
│
├── models/
│   └── baseline/                # Persisted vectorizer & baseline models
│       ├── metadata.json         # Experiment parameters & reproducibility config
│       ├── tfidf_vectorizer.joblib # TF-IDF vectorizer (274,360 features)
│       ├── logistic_regression.joblib # Logistic Regression model artifact
│       ├── linear_svc.joblib     # LinearSVC model artifact (Winning Baseline)
│       ├── cm_val_logistic_regression.png
│       ├── cm_val_linear_svc.png
│       └── cm_test_selected_baseline.png
│
├── app/                          # Streamlit UI placeholder
│
├── PROJECT_CONTEXT.md            # Living project context & status document
├── README.md                     # Setup and execution guide
├── requirements.txt              # Project dependencies
└── .gitignore                    # Version control rules
```

---

## ⚙️ Setup & Execution

### 1. Activate Virtual Environment & Install Dependencies

```bash
# Activate .venv (Windows PowerShell):
.venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

### 2. Execute Pipelines

#### Phase 1 Data Cleaning Pipeline:
```bash
python -m src.data.cleaning
```

#### Phase 2 Baseline Text Classifier Pipeline:
```bash
python -m src.models.baseline
```

### 3. Run Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_baseline_classifier.ipynb
```

---

## 📊 Baseline Classifier Benchmark Results (Phase 2)

- **Dataset**: `51,055` rows split into Train (`40,844`), Validation (`5,105`), Test (`5,106`).
- **Feature Extractor**: TF-IDF `(1, 2)` n-grams (`274,360` vocabulary features).

### Validation Set Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Selected Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7798 | 0.7349 | 0.7796 | |
| **LinearSVC** | **0.7843** | **0.7392** | **0.7821** | **Selected Winner** |

### Selected Baseline (LinearSVC) Test Set Performance
- **Test Accuracy**: `0.7763`
- **Test Macro F1**: `0.7165`
- **Test Weighted F1**: `0.7737`

---

## 🔮 Next Phase: Phase 3 — Fine-Tune DistilBERT Classifier

Fine-tune `distilbert-base-uncased` on `data/processed/splits/train.csv` and compare performance against the Phase 2 LinearSVC baseline (`Macro F1 = 0.7392`).
