# models/llm.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LABELS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

def load_llm(model_name="Qwen/Qwen2.5-7B-Instruct"):
    print("Loading LLM:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def prompt_llm(model, tokenizer, text: str) -> str:
    prompt = f"""
You are a multilabel classifier for toxic comments.

Possible labels:
{", ".join(LABELS)}

Instructions:
- Read the comment.
- Decide which labels apply.
- Answer with ONE short line containing only the labels separated by commas.
- If no label applies, answer: none

Examples:
Comment: "You are so stupid"
Assistant: insult

Comment: "I disagree with your opinion"
Assistant: none

Comment: "You f***ing idiot, I hate your kind"
Assistant: toxic, insult, identity_hate

Now classify the following comment.

Comment: {text}
Assistant:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.strip()


if __name__ == "__main__":
    model, tok = load_llm()
    result = prompt_llm(model, tok, "I hate you.")
    print(result)
