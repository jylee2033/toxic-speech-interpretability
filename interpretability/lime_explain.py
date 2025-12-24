# interpretability/lime_explain.py

from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )
    model.load_state_dict(torch.load("bert_jigsaw.pt"))
    model.eval()
    return model, tokenizer


def predict(texts, model, tokenizer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits).detach().numpy()
    return probs


def explain(text):
    model, tokenizer = load_bert()
    explainer = LimeTextExplainer(class_names=LABELS)
    explanation = explainer.explain_instance(
        text,
        lambda x: predict(x, model, tokenizer),
        num_features=10
    )
    explanation.save_to_file("lime_explanation.html")
    print("Saved lime_explanation.html")


if __name__ == "__main__":
    explain("I hate you so much")
