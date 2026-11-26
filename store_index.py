"""Create the Pinecone vector index from the repository's medical reference PDF."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_hugging_face_embeddings,
    load_pdf,
    text_split,
)

PINECONE_INDEX_NAME = "medicine"
PROJECT_ROOT = Path(__file__).resolve().parent


def create_vector_index():
    """Embed the bundled PDF chunks and store them in the Pinecone index."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("PINECONE_API_KEY")
    region = os.getenv("PINECONE_API_ENV")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in your environment or .env file.")
    if not region:
        raise RuntimeError("Set PINECONE_API_ENV in your environment or .env file.")

    documents = load_pdf(str(PROJECT_ROOT / "data"))
    chunks = text_split(documents)
    embeddings = download_hugging_face_embeddings()

    client = Pinecone(api_key=api_key)
    existing_indexes = [
        index.name for index in client.list_indexes().index_list["indexes"]
    ]
    if PINECONE_INDEX_NAME not in existing_indexes:
        client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )

    PineconeVectorStore.from_texts(
        [chunk.page_content for chunk in chunks],
        embeddings,
        index_name=PINECONE_INDEX_NAME,
    )


if __name__ == "__main__":
    create_vector_index()
