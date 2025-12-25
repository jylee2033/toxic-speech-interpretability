# Toxic Speech Interpretability

An interpretable toxic speech classification system that combines a fine-tuned BERT
model with post-hoc explanations (LIME, SHAP) and bias evaluation using human
rationales from the HateXplain dataset.


## Overview

This project studies **interpretability and fairness in toxic speech classification**.
While modern classifiers can achieve strong accuracy, their predictions are often
opaque, making it difficult to verify whether decisions rely on meaningful harmful
content or spurious cues such as identity terms.

We build a supervised BERT-based toxicity classifier, compare it against a zero-shot
large language model baseline, and evaluate model explanations using human-annotated
rationales.


## Motivation

- Toxicity classifiers are increasingly used in online moderation systems, where
  misclassification can cause real-world harm.
- Deep learning models may overfit to identity-related terms rather than genuinely
  harmful language.
- The HateXplain dataset provides human rationales, enabling systematic evaluation of
  interpretability and bias.

Our goal is to build a model that is **not only accurate, but also transparent and
diagnosable**.


## Approach

The system integrates four main components:

- **Supervised classification**  
  Fine-tuned `bert-base-uncased` on the Jigsaw Toxic Comment dataset for multi-label
  toxicity prediction.

- **Zero-shot LLM comparison**  
  Evaluation of a Qwen2.5-7B-Instruct model using instruction-based prompting, without
  task-specific fine-tuning.

- **Post-hoc interpretability**  
  Token-level explanations generated with:
  - LIME (local surrogate-based explanations)
  - SHAP (Shapley-value-based attributions)

- **Bias and interpretability evaluation**  
  Model-generated explanations are compared against human rationales from HateXplain
  using token-level alignment metrics.


## Datasets

- **Jigsaw Toxic Comment Classification Challenge**  
  Used for training and evaluating the BERT classifier.  
  Labels: toxic, severe toxic, obscene, threat, insult, identity hate.

- **HateXplain**  
  Used exclusively for interpretability evaluation.  
  Provides human-annotated rationales and target communities.


## Experimental Setup

- **Model**: BERT (`bert-base-uncased`)
- **Loss**: Binary cross-entropy (multi-label)
- **Training split**: 80 / 10 / 10 (train / validation / test)
- **Evaluation**:
  - Full test set for BERT
  - 200-sample subset for direct comparison with the zero-shot LLM
- **Interpretability metrics**:
  - Token-level Intersection-over-Union (IoU)
  - Precision / recall against human rationales


## Results

### Classification Performance

- Fine-tuned BERT macro F1 (full test set): **0.65**
- Fine-tuned BERT macro F1 (200-sample subset): **0.435**
- Zero-shot LLM macro F1 (200-sample subset): **0.18**

The supervised BERT model significantly outperforms the zero-shot LLM across all
toxicity categories.

### Interpretability

- LIME and SHAP consistently highlight abusive or hostile tokens in toxic samples.
- Non-toxic or polite sentences receive near-zero or negative token contributions.
- Mean IoU between LIME explanations and human rationales: **0.17**

These alignment scores are consistent with prior HateXplain benchmarks and reflect
known challenges in token-level plausibility evaluation.


## Repository Structure

preprocessing/      Dataset loading and preprocessing  
models/             BERT training and LLM interface  
interpretability/   LIME and SHAP explanation code  
evaluation/         Classification and interpretability evaluation  
output/             Generated explanation outputs and plots  
project.ipynb       Full project write-up and analysis


## Full Write-up

The complete experimental report, figures, and qualitative analysis are provided in
`project.ipynb`.


## Acknowledgements

This project was originally developed as part of **CMPT 413 (Natural Language Processing)**
at **Simon Fraser University**.


## Authors

- Janet Ahn  
- Jooyoung (Julia) Lee