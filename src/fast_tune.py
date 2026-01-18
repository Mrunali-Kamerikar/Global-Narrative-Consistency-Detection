import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from chunk import chunk_text
from retrieve import retrieve_chunks
from load_data import load_data

# -------------------------------
# Models
# -------------------------------
NLI_MODEL = "roberta-large-mnli"
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
model.eval()

# -------------------------------
# Load data
# -------------------------------
train, _, novels = load_data()

# Subsample for speed
N = min(200, len(train))
train = train.sample(N, random_state=42).reset_index(drop=True)

# -------------------------------
# Embed novels once
# -------------------------------
novel_chunks = {}
for book_name, text in novels.items():
    chunks = chunk_text(text)
    embs = EMB_MODEL.encode(chunks, show_progress_bar=True)
    novel_chunks[book_name] = list(zip(chunks, embs))

# -------------------------------
# MNLI contradiction
# -------------------------------
def contradiction_score(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return probs[0].item()   # 0 = contradiction

# -------------------------------
# Score each training example
# -------------------------------
scores = []
labels = []

for _, row in train.iterrows():
    backstory = row["content"]
    label = 0 if row["label"].lower() == "inconsistent" else 1
    book = row["book_name"]

    chunks = novel_chunks[book]
    query = EMB_MODEL.encode(backstory)

    sims = [(np.dot(query, emb), txt) for txt, emb in chunks]
    sims.sort(reverse=True)
    top_chunks = [t for _, t in sims[:5]]

    s = np.mean([contradiction_score(c, backstory) for c in top_chunks])

    scores.append(s)
    labels.append(label)

# -------------------------------
# Find best threshold
# -------------------------------
best_t = 0
best_acc = 0

for t in np.linspace(0.05, 0.95, 40):
    preds = [0 if s > t else 1 for s in scores]
    acc = np.mean([p == y for p, y in zip(preds, labels)])

    if acc > best_acc:
        best_acc = acc
        best_t = t

print("Best threshold:", best_t)
print("Train accuracy:", best_acc)

with open("best_threshold.txt", "w") as f:
    f.write(str(best_t))
