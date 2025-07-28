# ===============================
# Streamlit App - Toxic Comment Detection (Using .safetensors)
# ===============================
import streamlit as st
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizerFast

# ✅ Path to your saved model folder
MODEL_PATH = "bert-toxic-comment-model"  # Ensure model.safetensors is inside this folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Cache the model for faster loading
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        from_safetensors=True  # ✅ Explicitly load .safetensors weights
    )
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ✅ Toxic labels
label_cols = [
    'toxicity', 'severe_toxicity', 'obscene',
    'threat', 'insult', 'identity_attack', 'sexual_explicit'
]

# ✅ Prediction function
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

    # Results as dictionary
    results = {label: f"{prob:.2f}" for label, prob in zip(label_cols, probs)}
    return results

# ===============================
# ✅ Streamlit User Interface
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
