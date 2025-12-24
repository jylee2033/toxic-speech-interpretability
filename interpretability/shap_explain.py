# interpretability/shap_explain.py

import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def load_bert():
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load("bert_jigsaw.pt"))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


def explain(text):
    model, tokenizer = load_bert()

    def f(x):
        enc = tokenizer(list(x), return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**enc).logits
        return torch.sigmoid(logits).numpy()

    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))
    shap_values = explainer([text])
    shap.plots.text(shap_values[0])


if __name__ == "__main__":
    explain("This group of people is awful")
