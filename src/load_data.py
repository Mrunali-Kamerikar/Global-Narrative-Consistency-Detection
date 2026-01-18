import pandas as pd
import os

def normalize(name):
    return name.lower().strip()

def load_data():
    train = pd.read_csv("data/Dataset/train.csv")
    test = pd.read_csv("data/Dataset/test.csv")

    books_folder = "data/Dataset/Books"
    novels = {}

    for fname in os.listdir(books_folder):
        key = normalize(fname.replace(".txt", ""))

        path = os.path.join(books_folder, fname)
        with open(path, encoding="utf-8") as f:
            novels[key] = f.read()

    # Normalize book_name column in dataframes
    train["book_name"] = train["book_name"].apply(normalize)
    test["book_name"] = test["book_name"].apply(normalize)

    return train, test, novels
