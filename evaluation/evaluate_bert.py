# evaluation/evaluate_bert.py

import os
import sys

# Add project root to the path (needed for Colab)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

from transformers import AutoModelForSequenceClassification
from preprocessing.jigsaw import prepare_jigsaw_for_training, LABEL_COLS

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "bert_jigsaw.pt"
BATCH_SIZE = 64


def evaluate_bert():
    # Load new train/validation/test split
    dataset, tokenizer = prepare_jigsaw_for_training()
    test_ds = dataset["test"]
    print(f"Using {len(test_ds)} test samples for evaluation.")

    val_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fine-tuned BERT model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    preds_list = []
    truth_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            labels = batch["labels"].numpy()  # (batch, num_labels)

            logits = model(input_ids=input_ids, attention_mask=mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()  # (batch, num_labels)

            preds_list.append((probs > 0.5).astype(int))
            truth_list.append(labels.astype(int))

    # (num_batches, batch, num_labels) → (num_samples, num_labels)
    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(truth_list, axis=0)

    print("y_true shape:", y_true.shape)
    print("y_pred shape:", y_pred.shape)

    # Compute Macro F1 score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print("Macro F1:", macro_f1)


if __name__ == "__main__":
    evaluate_bert()
