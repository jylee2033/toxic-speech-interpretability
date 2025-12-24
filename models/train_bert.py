# models/train_bert.py

import os
import sys

# Add project root to the path (needed for Colab)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from preprocessing.jigsaw import prepare_jigsaw_for_training, LABEL_COLS

MODEL_NAME = "bert-base-uncased"
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16


def train_bert():
    dataset, tokenizer = prepare_jigsaw_for_training()
    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss:", total_loss / len(train_loader))

    torch.save(model.state_dict(), "bert_jigsaw.pt")
    print("Saved model → bert_jigsaw.pt")


if __name__ == "__main__":
    train_bert()
