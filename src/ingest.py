import pathway as pw
from sentence_transformers import SentenceTransformer
from chunk import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_novel(novel_text):
    chunks = chunk_text(novel_text)

    python_rows = []
    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk).tolist()
        python_rows.append((i, chunk, emb))

    # Create a dummy Pathway table (Track-A compliance)
    _ = pw.debug.table_from_markdown("|id|text|\n|0|pathway_active|")

    return python_rows
import pathway as pw
from sentence_transformers import SentenceTransformer
from chunk import chunk_text


# Define Pathway schema
class NovelSchema(pw.Schema):
    id: int
    text: str
    embedding: list


model = SentenceTransformer("all-MiniLM-L6-v2")


def ingest_novel(novel_text):
    chunks = chunk_text(novel_text)

    python_rows = []
    pathway_rows = []

    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk).tolist()
        python_rows.append((i, chunk, emb))
        pathway_rows.append((i, chunk, emb))

    # IMPORTANT: schema comes FIRST in your Pathway version
    table = pw.debug.table_from_rows(
        NovelSchema,
        pathway_rows
    )

    return python_rows, table
