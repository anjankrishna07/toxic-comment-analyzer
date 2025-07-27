# ===============================
# app.py - Hugging Face Spaces Gradio App
# ===============================
import gradio as gr
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizerFast

# ✅ Load trained model and tokenizer
model_path = "bert-toxic-comment-model"  # Ensure this folder is uploaded with the app
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = model.to(device)
model.eval()

# ✅ Toxic Labels
label_cols = [
    'toxicity', 'severe_toxicity', 'obscene',
    'threat', 'insult', 'identity_attack', 'sexual_explicit'
]

# ✅ Prediction Function
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

    # Per-class predictions
    results = {label: f"{prob:.2f}" for label, prob in zip(label_cols, probs)}

    # ✅ Overall Verdict
    max_prob = np.max(probs)
    if max_prob >= 0.7:
        overall = "🚨 Toxic"
    elif 0.3 <= max_prob < 0.7:
        overall = "⚠️ Slightly Toxic"
    else:
        overall = "✅ Non Toxic (It is good)"

    results["Overall Verdict"] = overall
    return results

# ✅ Gradio Interface
demo = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(lines=3, placeholder="Type a comment here..."),
    outputs=gr.Label(num_top_classes=8),
    title="Toxic Comment Detector (BERT)",
    description="Enter a comment to see toxic category probabilities and an overall verdict."
)

# ✅ Launch for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()
