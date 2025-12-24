# evaluation/evaluate_explains.py

import sys, os

# Add project root to the path (needed for Colab)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from collections.abc import Mapping
from collections import Counter

from preprocessing.hatexplain import load_hatexplain
from interpretability.lime_explain import load_bert, predict
from lime.lime_text import LimeTextExplainer


def iou(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0


def flatten_to_ints(x):
    """
    Flatten various rationale structures into a simple Python list of ints.
    """
    # If tensor → convert to list first
    if isinstance(x, torch.Tensor):
        x = x.tolist()

    # If dict → flatten values
    if isinstance(x, Mapping):
        res = []
        for v in x.values():
            res.extend(flatten_to_ints(v))
        return res

    # If list/tuple/set → recursively flatten each element
    if isinstance(x, (list, tuple, set)):
        res = []
        for v in x:
            res.extend(flatten_to_ints(v))
        return res

    # If single number → convert to int and wrap in list
    if isinstance(x, (int, np.integer)):
        return [int(x)]

    # Ignore other types
    return []


def normalize_rationales(ex, max_len=None):
    """
    Convert ex['rationales'] into a list of word indices.
    Optionally keep only indices within the first max_len words.
    """
    gold = ex["rationales"]
    idxs = flatten_to_ints(gold)

    if max_len is not None:
        idxs = [i for i in idxs if 0 <= i < max_len]

    return sorted(set(idxs))


def get_label(ex):
    """
    Extract the label from a tokenized HateXplain sample and return it as a Python scalar.
    """
    label = ex["label"]  # majority label from convert_to_dataframe()

    # Because of set_format("torch"), it may be a tensor
    if isinstance(label, torch.Tensor):
        label = label.item()

    return label


def evaluate():
    hate = load_hatexplain()
    model, tokenizer = load_bert()
    explainer = LimeTextExplainer()

    # Keep only samples that have at least one rationale
    hate = hate.filter(lambda ex: len(ex["rationales"]) > 0)
    print("Num examples with non-empty rationales:", len(hate))

    NUM_SAMPLES = 100

    samples = hate.select(range(NUM_SAMPLES))
    total = len(samples)

    per_class_scores = {} # store IoU scores per label
    all_scores = []

    for i, ex in enumerate(samples):
        print("=" * 80)
        print(f"[{i+1}/{total}] Running LIME...")

        # Prepare text (truncate to first 40 words)
        full_text = ex["text"]
        words = full_text.split()
        max_len = 40
        text_words = words[:max_len]
        text = " ".join(text_words)

        gold_indices = normalize_rationales(ex, max_len=max_len)

        # Run LIME explanation
        explanation = explainer.explain_instance(
            text,
            lambda x: predict(x, model, tokenizer),
            num_features=10,
            num_samples=500,
        )

        lime_words = [w for w, _ in explanation.as_list()]
        lime_set = {w.lower().strip() for w in lime_words}

        lime_indices = []
        for idx, w in enumerate(text_words):
            if w.lower().strip() in lime_set:
                lime_indices.append(idx)

        score = iou(lime_indices, gold_indices)
        all_scores.append(score)

        # Store score per class
        label = get_label(ex)
        if label not in per_class_scores:
            per_class_scores[label] = []
        per_class_scores[label].append(score)

    print("=" * 80)
    print("Mean IOU (overall):", np.mean(all_scores))
    print()

    # Print per-class IoU
    print("Per-class IOU:")
    for label, vals in per_class_scores.items():
        print(f"{label}: mean IoU={np.mean(vals):.4f}, count={len(vals)}")


if __name__ == "__main__":
    evaluate()
