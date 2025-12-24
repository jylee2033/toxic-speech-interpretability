# preprocessing/jigsaw.py

from datasets import load_dataset, Features, Sequence, Value, DatasetDict
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_jigsaw_dataset():
    return load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")

def preprocess_jigsaw(batch):
    texts = [str(t) for t in batch["comment_text"]]

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

    # Build labels as list of lists
    labels = [[batch[col][i] for col in LABEL_COLS] for i in range(len(texts))]
    encodings["labels"] = labels
    return encodings

def prepare_jigsaw_for_training():
    raw_dataset = load_jigsaw_dataset()  # Original HF dataset (train + test)

    # Use only HF training split, ignore HF test split
    full_train = raw_dataset["train"]

    # Manually split HF train split into train/val/test
    # First split: 80% train / 20% temporary set
    # Second split: temp → 50% val, 50% test
    tmp = full_train.train_test_split(test_size=0.2, seed=42)
    train_ds = tmp["train"]
    temp = tmp["test"]

    tmp2 = temp.train_test_split(test_size=0.5, seed=42)
    val_ds = tmp2["train"]
    test_ds = tmp2["test"]

    # Rebuild into a DatasetDict for easier use
    dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    # Define features for processed dataset
    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "token_type_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
        }
    )

    # Tokenize and encode labels for each split
    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            preprocess_jigsaw,
            batched=True,
            remove_columns=["comment_text", "id"] + LABEL_COLS,
            features=features,
        )

    dataset.set_format(type="torch")
    return dataset, tokenizer

if __name__ == "__main__":
    ds, tok = prepare_jigsaw_for_training()
    print(ds)
