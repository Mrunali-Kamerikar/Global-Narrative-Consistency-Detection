import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(backstory, novel_chunks, top_k=5):
    query = model.encode(backstory)

    scores = []
    for _, text, emb in novel_chunks:
        sim = np.dot(query, emb)
        scores.append((sim, text))

    scores.sort(reverse=True)
    return [text for _, text in scores[:top_k]]
