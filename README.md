# Simple Streamlit AI Document Assistant

An interactive document assistant built with Streamlit, Sentence Transformers, FAISS, and Groq API. It supports local file uploads as well as Google Drive document imports.

## Features
- **Supported File Types**: PDF, DOCX, TXT, MD.
- **Google Drive Integration**: Paste a Google Drive file or folder link to load documents.
- **Efficient Chunking & Processing**: Text is extracted, metadata (filename and page number) is retained, and embeddings are created once and saved in Streamlit's `session_state`.
- **Hybrid Search**: Combines FAISS vector similarity search and keyword matching to find relevant context.
- **Context-Bound QA**: Answers questions strictly using retrieved context via Groq API.
- **Sources Attribution**: Displays exact filenames, page numbers, and text chunks below every answer.

---

## Setup Instructions

### 1. Clone the repository and install dependencies
```bash
pip install -r requirements.txt