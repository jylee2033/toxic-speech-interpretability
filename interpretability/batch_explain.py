# interpretability/batch_explain.py

import os
import sys

# Add project root to the path (needed for Colab)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.jigsaw import LABEL_COLS

# Settings
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "bert_jigsaw.pt"

# toxic=0, severe_toxic=1, obscene=2, threat=3, insult=4, identity_hate=5
CLASS_IDX = 0  # Default: evaluate "toxic" class

# List of input texts
TEXT_LIST = [
    "I hate you so much.",
    "You are a disgusting idiot.",
    "Thank you for your help.",
    "I really appreciate your kindness.",
    "This is absolutely terrible work.",
    "You racist piece of garbage.",
]

LIME_DIR = "lime_batch_outputs"
SHAP_DIR = "shap_batch_outputs"
os.makedirs(LIME_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)


# Shared model / tokenizer / prediction helper
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device


def classify_fn(model, tokenizer, texts):
    """Prediction function used by LIME & SHAP: list[str] -> np.array of probabilities."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()  # (batch, num_labels)

    return probs


def main():
    print("Loading model & tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer()

    # Prepare LIME and SHAP explainers
    lime_explainer = LimeTextExplainer(class_names=LABEL_COLS)

    # SHAP text masker + explainer
    text_masker = shap.maskers.Text(tokenizer)
    shap_explainer = shap.Explainer(
        lambda x: classify_fn(model, tokenizer, x),
        text_masker,
        output_names=LABEL_COLS,
    )

    for i, text in enumerate(TEXT_LIST):
        print("=" * 80)
        print(f"[{i}] Text: {text}")
        print("→ Running LIME & SHAP...")

        # LIME
        lime_exp = lime_explainer.explain_instance(
            text,
            lambda x: classify_fn(model, tokenizer, x),
            labels=[CLASS_IDX],
            num_features=8,
            num_samples=500,
        )
        lime_path = os.path.join(LIME_DIR, f"lime_{i}_{LABEL_COLS[CLASS_IDX]}.html")
        lime_exp.save_to_file(lime_path)
        print(f"LIME saved to {lime_path}")

        # Compute SHAP values for single text sample
        shap_values = shap_explainer([text])  # shape: (1, tokens, num_labels)

        # Extract SHAP values for target class (CLASS_IDX)
        shap_for_label = shap_values[:, :, CLASS_IDX]  # (1, tokens)
        sv = shap_for_label[0]  # Explanation object for one example

        # Extract tokens and SHAP scores
        tokens = np.array(sv.data, dtype=object)
        scores = np.array(sv.values, dtype=float)

        # Remove empty or whitespace-only tokens
        mask = np.array([isinstance(t, str) and t.strip() != "" for t in tokens])
        tokens = tokens[mask]
        scores = scores[mask]

        # Select top_k tokens by absolute SHAP value
        top_k = min(10, len(tokens))
        idxs = np.argsort(-np.abs(scores))[:top_k]
        tokens_top = tokens[idxs]
        scores_top = scores[idxs]

        # Draw horizontal bar plot
        shap_path = os.path.join(SHAP_DIR, f"shap_{i}_{LABEL_COLS[CLASS_IDX]}.png")
        plt.figure()
        y_pos = np.arange(len(tokens_top))
        plt.barh(y_pos, scores_top)
        plt.yticks(y_pos, tokens_top)
        plt.xlabel("SHAP value")
        plt.title(f"SHAP ({LABEL_COLS[CLASS_IDX]}) for example {i}")
        plt.gca().invert_yaxis()  # Put the most important token at the top

        plt.tight_layout()
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
        print(f"SHAP saved to {shap_path}")

    print("\nDone! LIME →", LIME_DIR, ", SHAP →", SHAP_DIR)


if __name__ == "__main__":
    main()
