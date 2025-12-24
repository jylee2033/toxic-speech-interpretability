# preprocessing/hatexplain.py

import requests
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128

URL = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"


def load_hatexplain_raw():
    print("Downloading HateXplain...")
    return requests.get(URL).json()


def convert_to_dataframe(raw):
    rows = []
    for post_id, sample in raw.items():
        # majority label from annotators
        labels = [a["label"] for a in sample["annotators"]]
        majority = max(set(labels), key=labels.count)

        text = " ".join(sample["post_tokens"])

        # rationales are top-level, not per annotator
        spans = sample.get("rationales", [])

        # target communities
        targets = []
        for a in sample["annotators"]:
            targets.extend(a.get("target", []))

        rows.append({
            "post_id": post_id,
            "text": text,
            "label": majority,
            "rationales": spans,
            "targets": list(set(targets))
        })
    return pd.DataFrame(rows)



def tokenize(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    encoded = dataset.map(encode, batched=True)
    encoded.set_format("torch")
    return encoded


def load_hatexplain():
    raw = load_hatexplain_raw()
    df = convert_to_dataframe(raw)
    ds = Dataset.from_pandas(df)
    return tokenize(ds)


if __name__ == "__main__":
    hate = load_hatexplain()
    print(hate)
