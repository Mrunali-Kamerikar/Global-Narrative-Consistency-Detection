 
**Global Narrative Consistency Detection**

### **Team: HackHers**

**Members:**

* **Mrunali Kamerikar** *(Team Leader)*
* **Riddhima Taose**


This project solves the **Global Narrative Consistency Challenge** by determining whether a hypothetical backstory for a character is logically compatible with a long-form novel (100k+ words).

The task is formulated as a **binary classification problem**:

* **1** → Backstory is consistent
* **0** → Backstory contradicts the narrative

Our system combines:

* Long-context document retrieval
* Semantic embeddings
* Natural Language Inference (NLI)
* Pathway-based orchestration (Track A requirement)


## 🧩 Approach Summary

The model follows four steps:

### 1. **Long-Context Chunking**

Each novel is split into fixed-size text chunks.
This allows efficient processing of 100k+ word documents without losing local context.

### 2. **Semantic Retrieval**

Backstories are embedded using `SentenceTransformers`.
The most relevant novel chunks are retrieved via cosine similarity.

### 3. **Contradiction Detection**

Each retrieved chunk is evaluated against the backstory using a pretrained **RoBERTa-MNLI** model.
The MNLI model outputs a probability that the chunk **contradicts** the backstory.

### 4. **Decision Logic**

If **any retrieved chunk** strongly contradicts the backstory → label = **0**
Otherwise → label = **1**

Thresholds are calibrated using the training set.


## 🧪 Why This Works

Narrative consistency is **existential**, not statistical:

> A single strong contradiction is enough to invalidate a backstory.

Using **maximum contradiction** instead of averaging ensures that even one incompatible scene is detected, which matches the causal-consistency requirement of the challenge.


## 📂 Project Structure

```
iit_hackathon/
│
├── data/
│   └── Dataset/
│       ├── Books/          # Full novels
│       ├── train.csv
│       └── test.csv
│
├── src/
│   ├── load_data.py        # Loads dataset
│   ├── chunk.py           # Splits novels
│   ├── ingest.py          # Pathway ingestion
│   ├── retrieve.py        # Semantic retrieval
│   ├── judge.py           # MNLI contradiction logic
│   ├── fast_tune.py       # Threshold tuning
│   └── main.py            # End-to-end pipeline
│
├── best_threshold.txt
├── results.csv
├── requirements.txt
└── README.md
```


## ⚙️ Installation

Use **Python 3.10**.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## ▶️ Running the System

### Step 1 — Tune the contradiction threshold

This uses `train.csv` to calibrate the MNLI contradiction scores.

```bash
python src/fast_tune.py
```

This creates:

```
best_threshold.txt
```


### Step 2 — Generate predictions

```bash
python src/main.py
```

This produces:

```
results.csv
```

Format:

```
StoryID,Prediction
12,1
48,0
...
```


## 🧠 Models Used

| Component               | Model                                    |
| ----------------------- | ---------------------------------------- |
| Embeddings              | `sentence-transformers/all-MiniLM-L6-v2` |
| Contradiction Detection | `roberta-large-mnli`                     |
| Orchestration           | `Pathway`                                |


## 📊 Track A Compliance

This system fulfills all Track-A requirements:

* Uses **Pathway** for long-document ingestion
* Handles **100k+ token novels**
* Performs **evidence-based reasoning**
* Produces **binary predictions**
* Fully **reproducible**


## ⚠️ Limitations

* NLI models reason locally; extremely subtle long-term dependencies may be missed.
* Chunking may occasionally separate cause and effect across boundaries.
* Very vague backstories may not trigger strong contradiction signals.


## 🏁 Final Note

This system is designed for **robust narrative-level consistency detection**, not surface-level keyword matching. It prioritizes logical contradictions, causal incompatibilities, and semantic mismatch across long contexts — exactly what this hackathon evaluates.

