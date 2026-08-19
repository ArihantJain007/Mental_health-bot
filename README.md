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
- **Phase 3: Fine-tune DistilBERT Classifier** (In Progress — train on Colab GPU)

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
│   ├── 02_baseline_classifier.ipynb # Baseline classifier notebook
│   └── 03_distilbert_finetuning.ipynb # DistilBERT fine-tuning (Colab GPU)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── cleaning.py          # Modular dataset cleaning pipeline
│   └── models/
│       ├── __init__.py
│       ├── baseline.py          # Baseline modeling pipeline
│       └── distilbert_classifier.py # DistilBERT fine-tuning pipeline
│
├── models/
│   ├── baseline/                # Persisted vectorizer & baseline models
│   └── distilbert/              # DistilBERT checkpoints & best model (Colab output)
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

#### Phase 3 DistilBERT — Local Smoke Test (CPU-safe, no full training):
```bash
python -m src.models.distilbert_classifier --smoke-test
```

#### Phase 3 DistilBERT — Full Training (CUDA GPU only, e.g. Google Colab):
```bash
python -m src.models.distilbert_classifier --full-train \
    --data-dir data/processed/splits \
    --output-dir models/distilbert \
    --num-epochs 3 \
    --batch-size 16 \
    --lr 2e-5
```

Resume from the latest checkpoint:
```bash
python -m src.models.distilbert_classifier --full-train --resume \
    --data-dir data/processed/splits \
    --output-dir models/distilbert
```

### 3. Run Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_baseline_classifier.ipynb
jupyter notebook notebooks/03_distilbert_finetuning.ipynb
```

> **Important:** Run `03_distilbert_finetuning.ipynb` on **Google Colab with GPU**. Store Phase 2 split CSVs and all DistilBERT artifacts on **Google Drive** (`MyDrive/mental-health-chatbot/`).

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

## 🤖 Phase 3 — DistilBERT Fine-Tuning

Fine-tune `distilbert-base-uncased` on the **exact Phase 2 splits** with:

- `max_length=256`
- Class-weighted `CrossEntropyLoss` (train labels only)
- Validation **Macro F1** checkpoint selection
- Checkpoint save/resume
- Single held-out **test** evaluation after model selection
- Sample inference with predicted class and softmax probabilities

### Google Colab Full Training Command

After mounting Drive and verifying splits in `notebooks/03_distilbert_finetuning.ipynb`:

```bash
python -m src.models.distilbert_classifier --full-train \
    --data-dir /content/drive/MyDrive/mental-health-chatbot/data/processed/splits \
    --output-dir /content/drive/MyDrive/mental-health-chatbot/models/distilbert \
    --num-epochs 3 \
    --batch-size 16 \
    --lr 2e-5
```

DistilBERT test metrics will be recorded in `models/distilbert/metadata.json` after Colab training completes. Do not compare against baseline until those metrics exist.
