# ===============================
# STEP 7: STREAMLIT APP - TOXIC COMMENT DETECTION
# (Same logic as your Gradio version)
# ===============================
import streamlit as st
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizerFast

# ✅ Load trained model and tokenizer
MODEL_PATH = "bert-toxic-comment-model"  # Same as Gradio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ✅ Toxic labels (same order)
label_cols = [
    'toxicity', 'severe_toxicity', 'obscene',
    'threat', 'insult', 'identity_attack', 'sexual_explicit'
]

# ✅ Prediction Function (unchanged logic)
def predict_toxicity(comment):
    encoding = tokenizer(
        comment,
        add_special_tokens=True,
        truncation=True,
        max_length=64,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    results = {label: f"{prob:.2f}" for label, prob in zip(label_cols, probs)}
    return results

# ===============================
# Streamlit UI (Mirroring Gradio behavior)
# ===============================
st.title("🧪 Toxic Comment Detector (BERT)")
st.write("Enter a comment to see the predicted toxic category probabilities.")

user_input = st.text_area("Type a comment:")

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            results = predict_toxicity(user_input)

        st.subheader("Predicted Toxic Probabilities:")
        for label, prob in results.items():
            st.write(f"**{label}:** {prob}")
    else:
        st.warning("⚠️ Please type a comment before analyzing.")
