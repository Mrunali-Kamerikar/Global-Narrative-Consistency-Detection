import pandas as pd
import pathway as pw

from load_data import load_data
from ingest import ingest_novel
from retrieve import retrieve_chunks
from judge import judge_consistency


def main():
    print("Loading data...")
    train, test, novels = load_data()

    # Pre-embed each novel once
    print("Embedding novels...")
    novel_tables = {}

    for book_name, novel_text in novels.items():
        print(f"Embedding {book_name}...")
        chunks, table = ingest_novel(novel_text)
        novel_tables[book_name] = chunks

    # Run Pathway engine (compliance)
    pw.run()

    predictions = []

    for idx, row in test.iterrows():
        story_id = row["id"]
        backstory = row["content"]
        book_name = row["book_name"]

        # Select the correct novel
        novel_chunks = novel_tables[book_name]

        evidence = retrieve_chunks(backstory, novel_chunks)
        pred = judge_consistency(backstory, evidence)

        predictions.append([story_id, pred])

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test)}")

    df = pd.DataFrame(predictions, columns=["StoryID", "Prediction"])
    df.to_csv("results.csv", index=False)

    print("Saved results.csv")


if __name__ == "__main__":
    main()
