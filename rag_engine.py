import json
import os
import pickle
import shutil
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )


client = genai.Client(api_key=API_KEY)


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)


INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
DOCUMENTS_PATH = DATA_DIR / "documents.json"


# ============================================================
# GEMINI EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSION = 768


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_text(file_path):
    """
    Extract readable text from PDF, DOCX or TXT files.
    """

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        reader = PdfReader(str(file_path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):
            text = page.extract_text() or ""

            text = text.strip()

            if text:
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        return "\n\n".join(pages)


    if extension == ".docx":

        document = Document(str(file_path))

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)


    if extension == ".txt":

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )


    raise ValueError(
        "Unsupported file type. Supported formats: PDF, DOCX and TXT."
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic document text cleaning.
    """

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        cleaned = " ".join(line.split())

        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines).strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=150
):
    """
    Split text into overlapping chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_documents(texts):
    """
    Create document embeddings using Gemini.
    """

    if not texts:
        return []

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def embed_query(query):
    """
    Create a query embedding.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values


# ============================================================
# DOCUMENT METADATA
# ============================================================

def load_documents():

    if not DOCUMENTS_PATH.exists():
        return []

    try:

        with open(
            DOCUMENTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_documents(documents):

    with open(
        DOCUMENTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# FAISS
# ============================================================

def load_index():

    if not INDEX_PATH.exists():
        return None

    return faiss.read_index(
        str(INDEX_PATH)
    )


def load_chunks():

    if not CHUNKS_PATH.exists():
        return []

    try:

        with open(
            CHUNKS_PATH,
            "rb"
        ) as file:

            return pickle.load(file)

    except Exception:

        return []


def save_index(index, chunks):

    if index is not None:

        faiss.write_index(
            index,
            str(INDEX_PATH)
        )

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )


# ============================================================
# REBUILD ENTIRE KNOWLEDGE BASE
# ============================================================

def rebuild_index():

    documents = load_documents()

    all_chunks = []

    all_embeddings = []

    valid_documents = []


    for document in documents:

        filename = document["filename"]

        stored_path = (
            UPLOAD_DIR /
            document["stored_name"]
        )

        if not stored_path.exists():
            continue

        try:

            text = extract_text(
                stored_path
            )

            chunks = chunk_text(text)

            if not chunks:
                continue

            embeddings = embed_documents(
                chunks
            )

            for chunk in chunks:

                all_chunks.append(
                    {
                        "text": chunk,
                        "source": filename
                    }
                )

            all_embeddings.extend(
                embeddings
            )

            valid_documents.append(
                document
            )

        except Exception as error:

            print(
                f"Could not index {filename}: {error}"
            )


    if not all_embeddings:

        if INDEX_PATH.exists():
            INDEX_PATH.unlink()

        if CHUNKS_PATH.exists():
            CHUNKS_PATH.unlink()

        save_documents(
            valid_documents
        )

        return {
            "documents": len(valid_documents),
            "chunks": 0
        }


    vectors = np.array(
        all_embeddings,
        dtype="float32"
    )


    faiss.normalize_L2(
        vectors
    )


    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )


    index.add(vectors)


    save_index(
        index,
        all_chunks
    )


    save_documents(
        valid_documents
    )


    return {
        "documents": len(valid_documents),
        "chunks": len(all_chunks)
    }


# ============================================================
# ADD DOCUMENT
# ============================================================

def add_document(
    file_path,
    original_filename,
    stored_name
):
    """
    Add a new document to the knowledge base.
    """

    file_path = Path(file_path)

    text = extract_text(
        file_path
    )

    text = clean_text(text)

    if not text:
        raise ValueError(
            "No readable text was found in this document."
        )


    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "The document did not contain enough usable text."
        )


    embeddings = embed_documents(
        chunks
    )


    vectors = np.array(
        embeddings,
        dtype="float32"
    )


    faiss.normalize_L2(
        vectors
    )


    index = load_index()

    existing_chunks = load_chunks()


    if index is None:

        index = faiss.IndexFlatIP(
            vectors.shape[1]
        )


    index.add(vectors)


    for chunk in chunks:

        existing_chunks.append(
            {
                "text": chunk,
                "source": original_filename
            }
        )


    save_index(
        index,
        existing_chunks
    )


    documents = load_documents()


    documents.append(
        {
            "filename": original_filename,
            "stored_name": stored_name,
            "chunks": len(chunks)
        }
    )


    save_documents(
        documents
    )


    return {
        "filename": original_filename,
        "chunks_added": len(chunks),
        "total_chunks": len(existing_chunks)
    }


# ============================================================
# SEARCH
# ============================================================

def search_documents(
    query,
    top_k=5
):
    """
    Search the knowledge base using cosine similarity.
    """

    index = load_index()

    chunks = load_chunks()


    if (
        index is None
        or not chunks
        or index.ntotal == 0
    ):
        return []


    query_embedding = embed_query(
        query
    )


    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    faiss.normalize_L2(
        query_vector
    )


    k = min(
        top_k,
        index.ntotal
    )


    scores, indices = index.search(
        query_vector,
        k
    )


    results = []


    for score, position in zip(
        scores[0],
        indices[0]
    ):

        if position < 0:
            continue

        chunk = chunks[position]

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score)
            }
        )


    return results


# ============================================================
# DOCUMENT LIST
# ============================================================

def get_documents():

    return load_documents()


# ============================================================
# DELETE ONE DOCUMENT
# ============================================================

def delete_document(filename):

    documents = load_documents()

    target = None

    remaining = []


    for document in documents:

        if document["filename"] == filename and target is None:

            target = document

        else:

            remaining.append(
                document
            )


    if target is None:

        raise ValueError(
            "Document not found."
        )


    stored_path = (
        UPLOAD_DIR /
        target["stored_name"]
    )


    if stored_path.exists():

        stored_path.unlink()


    save_documents(
        remaining
    )


    result = rebuild_index()


    return result


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

def clear_index():

    if INDEX_PATH.exists():
        INDEX_PATH.unlink()


    if CHUNKS_PATH.exists():
        CHUNKS_PATH.unlink()


    if DOCUMENTS_PATH.exists():
        DOCUMENTS_PATH.unlink()


    if UPLOAD_DIR.exists():

        for item in UPLOAD_DIR.iterdir():

            try:

                if item.is_file() or item.is_symlink():
                    item.unlink()

                elif item.is_dir():
                    shutil.rmtree(item)

            except Exception as error:

                print(
                    f"Could not remove {item}: {error}"
                )


# ============================================================
# KNOWLEDGE STATUS
# ============================================================

def get_knowledge_status():

    documents = load_documents()

    chunks = load_chunks()

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "has_documents": bool(documents)
    }