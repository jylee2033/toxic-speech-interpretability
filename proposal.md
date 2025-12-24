# Title
Classifying Online Toxicity

## Abstract
Online platforms have become an undeniable means of connection in today’s age and they are often plagued with harassment, hate speech, and toxic language. In this project, we aim to analyze how a strong model makes predictions on toxic comments. Our primary focus will be on interpretability and fairness, understanding which words or patterns most influence the model’s toxicity predictions and whether the model exhibits biases against certain identity groups.

We will fine-tune a pre-trained BERT model on the Jigsaw Toxic Comment Classification dataset, which contains over 150,000 labeled Wikipedia comments across categories such as toxic, severe toxic, obscene, threat, insult, and identity hate. To interpret model predictions, we will use techniques such as LIME (Local Interpretable Model-Agnostic Explanations) and SHAP (Shapley Additive Explanations). These techniques provide human readable explanations, highlighting which words most contribute to a comment being classified as toxic, and allow us to detect potential systematic biases toward identity terms such as “Muslim,” “woman,” or “gay.”

Evaluation will focus on F1-score to ensure that the model performs reasonably while interpretability analysis is applied. By emphasizing explainability and fairness, this project aims to promote transparency in automated content moderation systems. Understanding how NLP models make decisions can help prevent unintended harms and ensure that moderation tools treat users fairly in online environments.