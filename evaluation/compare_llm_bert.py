# evaluation/compare_llm_bert.py

import sys
import os

# Add project root to the path (needed for Colab)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from sklearn.metrics import f1_score

import torch
from transformers import AutoModelForSequenceClassification

from models.llm import load_llm, prompt_llm
from preprocessing.jigsaw import prepare_jigsaw_for_training, LABEL_COLS


def parse_labels_from_response(resp: str):
    """
    Parse the last line of the LLM response.
    If it contains toxic, insult, etc., treat them as predicted labels.
    If 'none' appears, treat it as no labels.
    """
    if not resp:
        return None

    last = resp.strip().splitlines()[-1].lower()

    if "none" in last:
        return []

    labels = []
    for lab in LABEL_COLS:
        if lab in last:
            labels.append(lab)
    return labels


def compare_models():
    dataset, tokenizer = prepare_jigsaw_for_training()
    eval_set = dataset["test"].select(range(200))  # 200 samples

    # Build BERT ground truth by decoding text and collecting labels
    bert_truth = []
    texts = []

    for ex in eval_set:
        text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        texts.append(text)
        bert_truth.append(ex["labels"])

    # BERT predictions on the same 200 samples
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )
    bert_model.load_state_dict(torch.load("bert_jigsaw.pt", map_location="cpu"))
    bert_model.eval()

    bert_model.to("cuda" if torch.cuda.is_available() else "cpu")

    bert_preds = []

    for ex in eval_set:
        with torch.no_grad():
            inputs = {
                "input_ids": ex["input_ids"].unsqueeze(0),
                "attention_mask": ex["attention_mask"].unsqueeze(0),
            }
            if "token_type_ids" in ex:
                inputs["token_type_ids"] = ex["token_type_ids"].unsqueeze(0)

            inputs = {k: v.to(bert_model.device) for k, v in inputs.items()}

            logits = bert_model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            pred_arr = (probs > 0.5).astype(int)
            bert_preds.append(pred_arr)

    bert_f1 = f1_score(bert_truth, bert_preds, average="macro")
    print(f"BERT Macro F1 (on {len(eval_set)} samples): {bert_f1:.4f}")

    # LLM predictions on the same 200 samples
    model, llm_tokenizer = load_llm()

    llm_preds = []
    llm_truth = []

    for i, (text, truth_vec) in enumerate(zip(texts, bert_truth)):
        print(f"[LLM] {i+1}/{len(texts)}")
        resp = prompt_llm(model, llm_tokenizer, text)

        labels = parse_labels_from_response(resp)
        if labels is None:
            print("  -> parse failed, skipping this example.")
            continue

        pred_arr = [1 if l in labels else 0 for l in LABEL_COLS]
        llm_preds.append(pred_arr)
        llm_truth.append(truth_vec)

    if not llm_preds:
        print("No valid LLM predictions parsed; cannot compute F1.")
    else:
        llm_f1 = f1_score(llm_truth, llm_preds, average="macro")
        print(f"LLM Macro F1 (on {len(llm_preds)} valid samples): {llm_f1:.4f}")


if __name__ == "__main__":
    compare_models()
