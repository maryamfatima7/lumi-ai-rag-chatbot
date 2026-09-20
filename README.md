# ✨ Lumi AI

**Lumi AI** is an AI-powered document intelligence and knowledge assistant designed to let users upload documents, process their content, and interact with that information through an intelligent conversational interface.

It combines **Retrieval-Augmented Generation (RAG)** with document processing, vector search, and generative AI to provide context-aware answers based on uploaded knowledge.

---

## 🚀 Features

* 📄 **Document Upload**

  * Upload supported documents for AI-powered analysis.
  * Files are processed and stored in a dedicated upload directory.

* 🧠 **RAG-Powered Question Answering**

  * Retrieves relevant information from uploaded documents before generating responses.
  * Helps reduce irrelevant or unsupported responses.

* 🔎 **Vector Search**

  * Uses **FAISS** for efficient similarity search across document embeddings.

* 🤖 **Generative AI**

  * Uses Google's Gemini models to generate natural-language responses from retrieved context.

* 💬 **AI Chat Interface**

  * Ask questions about uploaded documents through a conversational interface.

* 🔐 **Supabase Integration**

  * Supabase is used for application data and authentication-related functionality.

* 📚 **Document Processing**

  * Supports document extraction and processing using Python-based libraries.

* ⚡ **Backend API**

  * Built with Flask and structured for deployment as a production web service.

* ☁️ **Deployment Ready**

  * Configured for cloud deployment with environment variables and production-oriented settings.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Lumi AI UI     │
                    │   Chat + Documents  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Flask API       │
                    │   Backend Server    │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌─────────────────┐          ┌─────────────────┐
       │ Document Upload │          │    Supabase     │
       │ & Processing    │          │ Auth / Data     │
       └────────┬────────┘          └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │ Text Extraction │
       │ & Chunking      │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │    Embeddings   │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │     FAISS       │
       │  Vector Search  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Gemini LLM      │
       │ Response Gen.   │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Context-Aware   │
       │ AI Response     │
       └─────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* REST API
* python-dotenv

### AI / RAG

* Google Gemini
* Retrieval-Augmented Generation (RAG)
* Embeddings
* FAISS
* Sentence Transformers

### Document Processing

* PyPDF
* python-docx

### Database / Authentication

* Supabase

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Railway
* Environment-based configuration

---

## 📁 Project Structure

```text
Lumi-AI/
│
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── uploads/
    └── ...
```

> The upload directory can also be configured through the `LUMI_UPLOAD_DIR` environment variable.

---

## ⚙️ Environment Variables

Create a `.env` file locally and configure the required credentials:

```env
GEMINI_API_KEY=your_gemini_api_key

SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

LUMI_UPLOAD_DIR=/tmp/lumi_ai_data/uploads
```

**Never commit `.env` or API keys to GitHub.**

---

## 💻 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/maryamfatima7/Lumi-AI.git
cd Lumi-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create `.env` and add your API credentials.

### 6. Run the application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 🔄 How Lumi AI Works

### 1. Upload

The user uploads a supported document through the Lumi AI interface.

### 2. Extract

The backend extracts readable text from the document.

### 3. Chunk

Large documents are divided into smaller text chunks suitable for retrieval.

### 4. Embed

Text chunks are converted into vector representations.

### 5. Index

The vectors are stored/indexed using FAISS for similarity search.

### 6. Retrieve

When the user asks a question, Lumi AI searches the vector index for the most relevant document content.

### 7. Generate

The retrieved context is provided to the Gemini model to generate a relevant response.

### 8. Respond

The final context-aware answer is returned to the user through the chat interface.

---

## 🧠 RAG Pipeline

```text
Document
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Vector Index
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
Gemini
   ↓
AI Response
```

---

## 🔐 Security

The project follows environment-based configuration for sensitive credentials.

Sensitive information such as:

* API keys
* Supabase credentials
* Environment secrets

should never be committed to the repository.

The `.gitignore` should include files such as:

```text
.env
venv/
__pycache__/
*.pyc
```

---

## ☁️ Deployment

Lumi AI is designed to be deployable to cloud platforms such as **Railway**.

For deployment:

1. Connect the GitHub repository.
2. Configure the required environment variables.
3. Install dependencies from `requirements.txt`.
4. Configure the production start command.
5. Deploy the backend.
6. Verify the health/API endpoints.
7. Test document upload and AI responses.

---

## 🎯 Project Goals

Lumi AI was built to explore practical AI engineering concepts including:

* Retrieval-Augmented Generation
* Vector databases and similarity search
* LLM integration
* Document intelligence
* API development
* Authentication and data services
* Cloud deployment
* Environment-based configuration

The project focuses on building an AI system that works with **user-provided knowledge rather than relying only on general model knowledge**.

---

## 🔮 Future Improvements

Potential future enhancements include:

* Multi-document knowledge bases
* Conversation memory
* Streaming AI responses
* Improved citation/source tracking
* Advanced document formats
* User-specific knowledge collections
* Background document processing
* Production-grade vector database
* More granular access control
* AI-powered document summarization

---

## 👩‍💻 Author

**Maryam Fatima**

AI Engineering & Full-Stack Development Learner

GitHub: `maryamfatima7`

---

## 📌 Disclaimer

Lumi AI is a learning and portfolio project demonstrating practical implementation of AI, RAG, document processing, backend APIs, and cloud deployment concepts.
