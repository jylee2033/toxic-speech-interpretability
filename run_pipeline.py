# run_pipeline.py

import os
from models import train_bert
from evaluation import evaluate_bert, compare_llm_bert, evaluate_explains
from interpretability import lime_explain, shap_explain

def main():
    print("=== Step 1: Train baseline BERT on Jigsaw ===")
    if not os.path.exists("bert_jigsaw.pt"):
        train_bert.train_bert()
    else:
        print("Found bert_jigsaw.pt, skipping training.")

    print("=== Step 2: Evaluate BERT performance ===")
    evaluate_bert.evaluate_bert()

    print("=== Step 3: Prompt LLM for comparison ===")
    compare_llm_bert.compare_models()

    print("=== Step 4: Interpret predictions ===")
    lime_explain.explain("I hate you so much")
    shap_explain.explain("This group of people is awful")

    print("=== Step 5: Evaluate explanations against HateXplain ===")
    evaluate_explains.evaluate()

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()