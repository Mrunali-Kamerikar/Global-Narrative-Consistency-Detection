from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "roberta-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

THRESHOLD = float(open("best_threshold.txt").read())

def contradiction_score(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]

    # roberta-large-mnli:
    # 0 = contradiction, 1 = neutral, 2 = entailment
    return probs[0].item()


def judge_consistency(backstory, evidence_chunks):
    if len(evidence_chunks) == 0:
        return 1

    scores = []
    for chunk in evidence_chunks:
        scores.append(contradiction_score(chunk, backstory))

    max_contradiction = max(scores)

    if max_contradiction > THRESHOLD:
        return 0
    else:
        return 1

