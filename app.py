"""Flask chat application backed by a Pinecone retrieval index and local Llama model."""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "medicine"
MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    str(Path(__file__).resolve().parent / "model" / "llama-2-7b-chat.ggmlv3.q4_0.bin"),
)

if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in your environment or .env file.")

# Initialize Pinecone and the existing vector index.
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings_model = download_hugging_face_embeddings()
vector_store = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, embeddings_model
)

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
llm = CTransformers(
    model=MODEL_PATH,
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.8},
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


@app.route("/")
def index():
    """Render the chatbot page."""
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Answer a submitted question using retrieved PDF context."""
    question = request.values.get("msg", "").strip()
    if not question:
        return "Please enter a question.", 400

    result = qa.invoke({"query": question})
    return str(result["result"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
