import os
import json
import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast

# Optional Gemini SDK import with graceful fallback
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration & Relative Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DISTILBERT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert")
BEST_MODEL_DIR = os.path.join(DISTILBERT_DIR, "best_model")
TOKENIZER_DIR = os.path.join(DISTILBERT_DIR, "tokenizer")
LABEL_MAPPING_PATH = os.path.join(DISTILBERT_DIR, "label_mapping.json")

DEFAULT_MAX_LENGTH = 256

FALLBACK_CANONICAL_LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]


# -----------------------------------------------------------------------------
# Cached Model & Tokenizer Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifier_artifacts():
    """
    Loads model, tokenizer, and canonical label mappings from local disk.
    Cached across user interactions for maximum efficiency.
    """
    if not os.path.exists(BEST_MODEL_DIR):
        raise FileNotFoundError(
            f"Trained model directory not found at: '{BEST_MODEL_DIR}'. "
            "Please ensure DistilBERT training has completed and best_model artifacts are present."
        )

    tokenizer_path = TOKENIZER_DIR if os.path.exists(TOKENIZER_DIR) else BEST_MODEL_DIR
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer directory not found at: '{tokenizer_path}'. "
            "Please check model artifact structure under 'models/distilbert/'."
        )

    label2id = {}
    id2label = {}
    if os.path.exists(LABEL_MAPPING_PATH):
        try:
            with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
                label2id = mapping_data.get("label2id", {})
                raw_id2label = mapping_data.get("id2label", {})
                id2label = {int(k): v for k, v in raw_id2label.items()}
        except Exception as e:
            st.warning(f"Could not load '{LABEL_MAPPING_PATH}': {e}. Falling back to default canonical labels.")

    if not label2id or not id2label:
        label2id = {label: i for i, label in enumerate(FALLBACK_CANONICAL_LABELS)}
        id2label = {i: label for i, label in enumerate(FALLBACK_CANONICAL_LABELS)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        BEST_MODEL_DIR,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()

    return model, tokenizer, label2id, id2label


# -----------------------------------------------------------------------------
# DistilBERT Classification Logic
# -----------------------------------------------------------------------------
def classify_statement(
    statement: str,
    model,
    tokenizer,
    id2label: dict,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """
    Tokenizes input statement, runs forward pass on CPU/GPU, applies softmax,
    and returns predicted category, confidence score, and full probability distribution.
    Does NOT modify model state.
    """
    if not statement or not statement.strip():
        return None

    inputs = tokenizer(
        statement,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    if probs.ndim == 0:
        probs = np.array([probs.item()])

    pred_id = int(np.argmax(probs))
    pred_label = id2label.get(pred_id, f"Category {pred_id}")
    pred_confidence = float(probs[pred_id])

    probabilities = {id2label.get(i, f"Class {i}"): float(probs[i]) for i in range(len(probs))}

    return {
        "statement": statement,
        "predicted_label": pred_label,
        "confidence": pred_confidence,
        "probabilities": probabilities,
    }


# -----------------------------------------------------------------------------
# Gemini Conversational Generation Logic
# -----------------------------------------------------------------------------
def get_gemini_api_key(user_key: str = "") -> str | None:
    """Retrieves Gemini API key from user UI input, environment variables, or Streamlit secrets."""
    if user_key and user_key.strip():
        return user_key.strip()

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
        if secret_key and str(secret_key).strip():
            return str(secret_key).strip()
    except Exception:
        pass

    return None


def generate_empathetic_response(
    statement: str,
    predicted_category: str,
    confidence: float,
    api_key: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Generates a compassionate, non-diagnostic response using Gemini API
    contextualized by DistilBERT's predicted category.

    Returns: (response_text, error_message)
    """
    if not GENAI_AVAILABLE:
        return None, "The 'google-genai' SDK is not installed. Run 'pip install google-genai'."

    key = get_gemini_api_key(api_key)
    if not key:
        return None, (
            "Gemini API Key is missing. Please set the 'GEMINI_API_KEY' environment variable, "
            "add it to Streamlit secrets, or enter your API key in the sidebar."
        )

    try:
        client = genai.Client(api_key=key)

        prompt = f"""
User Statement: "{statement}"
Classifier Context: DistilBERT predicted mental health category: "{predicted_category}" (Confidence: {confidence * 100:.1f}%).

Instructions:
- Write an empathetic, supportive, and compassionate response to the user statement.
- Validate their feelings in the context of the identified mental health category ({predicted_category}).
- DO NOT provide any medical diagnosis, clinical assessment, or treatment prescriptions.
- DO NOT attempt to re-classify the statement or debate the classifier result.
- Keep the tone warm, respectful, supportive, and conversational.
- If the statement or category indicates crisis or self-harm ('Suicidal'), include compassionate crisis guidance: "If you may be in immediate danger or think you might hurt yourself, contact your local emergency services or a crisis service available in your country, and reach out to someone you trust who can stay with you."
"""

        system_instruction = (
            "You are an empathetic, compassionate AI Mental Health Support assistant. "
            "You provide supportive, non-diagnostic conversational responses to help users feel heard and validated."
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_instruction,
            ),
        )

        if response and response.text:
            return response.text.strip(), None
        else:
            return None, "Gemini API returned an empty response."

    except Exception as e:
        return None, f"Gemini API Error: {str(e)}"


# -----------------------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Mental Health Support Chatbot",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for polished, modern layout
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .disclaimer-box {
            background-color: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 0.85rem 1.1rem;
            border-radius: 0.375rem;
            font-size: 0.9rem;
            color: #334155;
            margin-bottom: 1.5rem;
        }
        .result-card {
            background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 100%);
            border: 1px solid #BFDBFE;
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .category-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #1E40AF;
            font-weight: 600;
        }
        .category-name {
            font-size: 1.8rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-top: 0.2rem;
        }
        .confidence-badge {
            display: inline-block;
            background-color: #2563EB;
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            margin-top: 0.5rem;
        }
        .ai-response-box {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            color: #0F172A !important;
            font-size: 1rem;
            line-height: 1.6;
        }
        .ai-response-box *, .ai-response-box p, .ai-response-box div, .ai-response-box span {
            color: #0F172A !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Options & API Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        user_api_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            help="Enter key if not set in GEMINI_API_KEY environment variable or Streamlit secrets.",
        )
        st.markdown("---")
        st.header("ℹ️ Architecture")
        st.markdown(
            """
            - **Classifier**: Fine-tuned `DistilBERT` (Local)
            - **Response Generator**: `Gemini API`
            - **Target Categories (7)**:
              Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal
            """
        )
        st.caption("Phase 4 — Step 2: DistilBERT + Gemini Pipeline")

    # Header & Title
    st.markdown('<div class="main-header">🧠 AI Mental Health Support Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empathetic Conversational Support Powered by DistilBERT & Gemini</div>', unsafe_allow_html=True)

    # Non-diagnostic disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            🛡️ <strong>Non-Diagnostic Disclaimer:</strong> This application uses a fine-tuned NLP classifier 
            to understand user statements and generate compassionate, non-diagnostic responses. 
            It is <strong>not</strong> a medical or diagnostic tool. If you may be in immediate danger or think you might hurt yourself, 
            contact your local emergency services or a crisis service available in your country, and reach out to someone you trust who can stay with you.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Model & Tokenizer with clear error handling
    try:
        model, tokenizer, label2id, id2label = load_classifier_artifacts()
    except FileNotFoundError as fnf_error:
        st.error(f"❌ **Artifact Loading Error:** {fnf_error}")
        st.info(
            "💡 **Help:** Ensure trained model files are in `models/distilbert/best_model/` "
            "and tokenizer in `models/distilbert/tokenizer/` relative to project root."
        )
        return
    except Exception as exc:
        st.error(f"❌ **Unexpected Error Loading Model:** {exc}")
        return

    # User Input Section
    st.subheader("Share Your Thoughts")

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    preset_text = ""
    if sample_col1.button("Sample: Anxiety"):
        preset_text = "I feel deeply anxious and overwhelmed by everything today."
    elif sample_col2.button("Sample: Depression"):
        preset_text = "I have been feeling really sad, hopeless, and exhausted lately."
    elif sample_col3.button("Sample: Normal"):
        preset_text = "I had a productive day at work and enjoyed spending time with my family."

    user_statement = st.text_area(
        label="Input Text",
        value=preset_text,
        placeholder="Type how you are feeling or share your statement here...",
        height=120,
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 3])
    submitted = col_btn.button("💙 Send Statement", type="primary", use_container_width=True)

    # Execute End-to-End Pipeline
    if (submitted or preset_text) and user_statement.strip():
        # Step 1: DistilBERT Classification
        with st.spinner("Classifying statement with DistilBERT..."):
            clf_result = classify_statement(user_statement, model, tokenizer, id2label)

        if clf_result:
            pred_label = clf_result["predicted_label"]
            confidence = clf_result["confidence"]
            probs = clf_result["probabilities"]

            # Display Classification Context Card
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="category-title">Classifier Context</div>
                    <div class="category-name">{pred_label}</div>
                    <div class="confidence-badge">Confidence: {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # High-risk crisis notice for Suicidal prediction (location-neutral)
            if pred_label == "Suicidal":
                st.error(
                    "⚠️ **Crisis Support Notice:** If you may be in immediate danger or think you might hurt yourself, "
                    "contact your local emergency services or a crisis service available in your country, "
                    "and reach out to someone you trust who can stay with you."
                )

            # Step 2: Gemini Conversational Response Generation
            with st.spinner("Generating empathetic response with Gemini..."):
                response_text, error_msg = generate_empathetic_response(
                    statement=user_statement,
                    predicted_category=pred_label,
                    confidence=confidence,
                    api_key=user_api_key,
                )

            if response_text:
                st.subheader("💬 AI Support Response")
                st.markdown(
                    f"""
                    <div class="ai-response-box">
                        {response_text.replace('\n', '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif error_msg:
                st.warning(f"⚠️ **Conversational Response Unavailable:** {error_msg}")

            # Expandable Classification Details (fulfills requirement 11: non-intrusive probability breakdown)
            with st.expander("📊 View Model Classification Details"):
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for cat_name, prob_val in sorted_probs:
                    prob_pct = prob_val * 100
                    is_top = (cat_name == pred_label)
                    c_name, c_bar, c_pct = st.columns([2.5, 5, 1.5])
                    with c_name:
                        if is_top:
                            st.markdown(f"**👉 {cat_name}**")
                        else:
                            st.write(cat_name)
                    with c_bar:
                        st.progress(float(prob_val))
                    with c_pct:
                        if is_top:
                            st.markdown(f"**{prob_pct:.2f}%**")
                        else:
                            st.write(f"{prob_pct:.2f}%")


if __name__ == "__main__":
    main()
