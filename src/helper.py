"""Utilities for preparing PDF text and sentence embeddings."""

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf(data_directory):
    """Load every PDF in the supplied directory as LangChain documents."""
    loader = DirectoryLoader(
        data_directory,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def text_split(extracted_data):
    """Split loaded documents into 500-character chunks with 20-character overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    """Load the sentence-transformer model used to embed documents and questions."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
