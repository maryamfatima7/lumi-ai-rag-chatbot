import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your .env file."
    )


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=API_KEY
)

CHAT_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite"
]

MAX_RAG_CONTEXT_CHARS = 30000


# ============================================================
# RAG
# ============================================================

from rag_engine import (
    add_document,
    search_documents,
    clear_index,
    get_knowledge_status,
    get_documents,
    delete_document,
    UPLOAD_DIR
)


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_INSTRUCTION = """
You are Lumi AI, a professional, friendly and intelligent AI
assistant.

You can operate in two modes:

1. Normal AI mode
2. Retrieval-Augmented Generation (RAG) mode

GENERAL RULES:

- Answer clearly and naturally.
- Be helpful and accurate.
- Use concise paragraphs and bullet points when useful.
- Do not invent facts.
- If the user asks about uploaded documents, use the supplied
  document context.
- Never claim that something came from a document unless it is
  actually present in the supplied context.
- If relevant document context is missing, be honest about it.
- You are called Lumi AI.
"""


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_rag_context(rag_results):

    if not rag_results:
        return ""

    context_parts = []
    total_chars = 0

    for item in rag_results:

        if not isinstance(item, dict):
            continue

        source = safe_text(
            item.get(
                "source",
                "Unknown document"
            )
        )

        text = safe_text(
            item.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        try:
            score = float(
                item.get(
                    "score",
                    0
                )
            )
        except (TypeError, ValueError):
            score = 0.0

        remaining = (
            MAX_RAG_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        text = text[:remaining]

        block = (
            f"SOURCE: {source}\n"
            f"SIMILARITY: {score:.3f}\n\n"
            f"{text}"
        )

        context_parts.append(block)

        total_chars += len(block)

    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# GEMINI
# ============================================================

def call_gemini(prompt):

    last_error = None

    for model in CHAT_MODELS:

        try:

            print()
            print("=" * 60)
            print(
                f"GEMINI REQUEST: {model}"
            )
            print("=" * 60)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2048
                )
            )

            answer = safe_text(
                getattr(
                    response,
                    "text",
                    ""
                )
            )

            if answer:

                print(
                    f"GEMINI SUCCESS: {model}"
                )

                return answer

            last_error = RuntimeError(
                f"{model} returned an empty response."
            )

            print(
                f"GEMINI EMPTY RESPONSE: {model}"
            )

        except Exception as error:

            last_error = error

            print(
                f"GEMINI ERROR ({model}):",
                repr(error)
            )

            error_text = str(error).lower()

            is_temporary = any(
                keyword in error_text
                for keyword in [
                    "503",
                    "unavailable",
                    "overloaded",
                    "high demand",
                    "temporarily",
                    "429",
                    "resource exhausted",
                    "timeout",
                    "timed out"
                ]
            )

            if is_temporary:

                print(
                    f"{model} is temporarily "
                    "unavailable."
                )

                print(
                    "Trying next Gemini model..."
                )

                continue

            raise

    raise RuntimeError(
        "All configured Gemini models are "
        "temporarily unavailable. "
        f"Last error: {last_error}"
    )


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    message,
    history,
    rag_context=None
):

    history = history or []

    conversation_parts = []

    for item in history[-12:]:

        if not isinstance(
            item,
            dict
        ):
            continue

        role = safe_text(
            item.get(
                "role",
                ""
            )
        )

        content = safe_text(
            item.get(
                "content",
                ""
            )
        )

        if not content:
            continue

        if role == "user":

            conversation_parts.append(
                f"User: {content}"
            )

        elif role == "assistant":

            conversation_parts.append(
                f"Lumi AI: {content}"
            )

    conversation = "\n\n".join(
        conversation_parts
    )

    document_context = (
        build_rag_context(
            rag_context
        )
    )

    if document_context:

        prompt = f"""
{SYSTEM_INSTRUCTION}

You are currently answering a question with RAG support.

RELEVANT DOCUMENT CONTEXT:

==============================
{document_context}
==============================

PREVIOUS CONVERSATION:

{conversation}

CURRENT USER MESSAGE:

{message}

RAG INSTRUCTIONS:

- Use relevant document context when answering.
- Prefer information from the documents when the question is
  about those documents.
- Do not invent information that is not supported by the context.
- If the answer cannot be found in the supplied document context,
  clearly say that you could not find the answer in the uploaded
  documents.
- If the question is general and unrelated to the documents,
  answer normally.
"""

    else:

        prompt = f"""
{SYSTEM_INSTRUCTION}

PREVIOUS CONVERSATION:

{conversation}

CURRENT USER MESSAGE:

{message}

Answer naturally and helpfully.
"""

    return call_gemini(
        prompt
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "success": True,
            "app": "Lumi AI",
            "status": "online"
        }
    )


# ============================================================
# CHAT
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        message = safe_text(
            data.get(
                "message",
                ""
            )
        )

        history = data.get(
            "history",
            []
        )

        rag_enabled = bool(
            data.get(
                "rag_enabled",
                False
            )
        )

        if not message:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Please enter a message."
                    )
                }
            ), 400

        rag_results = []

        if rag_enabled:

            try:

                print(
                    "RAG: searching documents..."
                )

                rag_results = (
                    search_documents(
                        message,
                        top_k=5
                    )
                    or []
                )

                print(
                    "RAG: found",
                    len(rag_results),
                    "results"
                )

            except Exception as rag_error:

                print(
                    "RAG SEARCH ERROR:",
                    repr(rag_error)
                )

                # Keep the chatbot usable even if
                # the RAG search has an issue.
                rag_results = []

        answer = generate_response(
            message=message,
            history=history,
            rag_context=rag_results
        )

        sources = []

        for item in rag_results:

            if not isinstance(
                item,
                dict
            ):
                continue

            source = safe_text(
                item.get(
                    "source",
                    ""
                )
            )

            if (
                source
                and source not in sources
            ):

                sources.append(
                    source
                )

        return jsonify(
            {
                "success": True,
                "answer": answer,
                "sources": sources,
                "rag_used": bool(
                    rag_results
                )
            }
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("CHAT ERROR")
        print("=" * 60)
        print(
            repr(error)
        )
        print("=" * 60)
        print()

        error_text = str(error).lower()

        temporary_error = any(
            keyword in error_text
            for keyword in [
                "503",
                "unavailable",
                "overloaded",
                "high demand",
                "temporarily",
                "429",
                "resource exhausted",
                "timeout",
                "timed out"
            ]
        )

        if temporary_error:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Lumi AI is temporarily "
                        "unavailable. Please try again."
                    )
                }
            ), 503

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_document():

    try:

        if "file" not in request.files:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No file was uploaded."
                    )
                }
            ), 400

        file = request.files["file"]

        if not file.filename:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Please select a file."
                    )
                }
            ), 400

        original_filename = Path(
            file.filename
        ).name

        extension = Path(
            original_filename
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Only PDF, DOCX and TXT "
                        "files are supported."
                    )
                }
            ), 400

        unique_name = (
            f"{uuid.uuid4().hex}"
            f"{extension}"
        )

        saved_path = (
            UPLOAD_DIR /
            unique_name
        )

        file.save(
            saved_path
        )

        print(
            "UPLOAD:",
            original_filename
        )

        result = add_document(
            saved_path,
            original_filename,
            unique_name
        )

        return jsonify(
            {
                "success": True,
                "message": (
                    f"{original_filename} "
                    "was successfully added "
                    "to Lumi AI."
                ),
                "document": result
            }
        )

    except Exception as error:

        print(
            "UPLOAD ERROR:",
            repr(error)
        )

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# KNOWLEDGE STATUS
# ============================================================

@app.route(
    "/api/knowledge",
    methods=["GET"]
)
def knowledge():

    try:

        status = (
            get_knowledge_status()
        )

        return jsonify(
            {
                "success": True,
                **status
            }
        )

    except Exception as error:

        print(
            "KNOWLEDGE ERROR:",
            repr(error)
        )

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.route(
    "/api/documents",
    methods=["GET"]
)
def documents():

    try:

        return jsonify(
            {
                "success": True,
                "documents": get_documents()
            }
        )

    except Exception as error:

        print(
            "DOCUMENT LIST ERROR:",
            repr(error)
        )

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.route(
    "/api/documents/<path:filename>",
    methods=["DELETE"]
)
def remove_document(filename):

    try:

        result = delete_document(
            filename
        )

        return jsonify(
            {
                "success": True,
                "message": (
                    f"{filename} was removed."
                ),
                "status": result
            }
        )

    except Exception as error:

        print(
            "DELETE DOCUMENT ERROR:",
            repr(error)
        )

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

@app.route(
    "/api/knowledge/clear",
    methods=["DELETE"]
)
def clear_knowledge():

    try:

        clear_index()

        return jsonify(
            {
                "success": True,
                "message": (
                    "Lumi AI knowledge base "
                    "was cleared."
                )
            }
        )

    except Exception as error:

        print(
            "CLEAR ERROR:",
            repr(error)
        )

        return jsonify(
            {
                "success": False,
                "error": str(error)
            }
        ), 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify(
        {
            "success": False,
            "error": (
                "File is too large. "
                "Maximum allowed size is 10 MB."
            )
        }
    ), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("                    LUMI AI")
    print("              AI CHATBOT + RAG")
    print("=" * 60)
    print()

    print(
        "Models:",
        ", ".join(CHAT_MODELS)
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "RAG: Enabled"
    )

    print(
        "Maximum upload size: 10 MB"
    )

    print()
    print(
        "Press CTRL+C to stop the server."
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )