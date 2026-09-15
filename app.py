import os
import re
import tempfile
import numpy as np
import streamlit as st
import pypdf
import docx
import gdown
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set Streamlit page configuration
st.set_page_config(page_title="AI Document Assistant", layout="wide")

# ==========================================
# 1. INITIALIZATION & CACHING
# ==========================================

@st.cache_resource
def load_embedding_model():
    # Load SentenceTransformer model once
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Initialize session state for storing documents and embeddings
if "chunks" not in st.session_state:
    st.session_state.chunks = []  # List of dicts: {"text": str, "filename": str, "page": int/str}
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None  # Numpy array of embeddings
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

# ==========================================
# 2. DOCUMENT EXTRACTION FUNCTIONS
# ==========================================

def extract_pdf(file_stream, filename):
    chunks = []
    reader = pypdf.PdfReader(file_stream)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append({
                "text": text.strip(),
                "filename": filename,
                "page": page_num
            })
    return chunks

def extract_docx(file_stream, filename):
    doc = docx.Document(file_stream)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [{"text": full_text, "filename": filename, "page": "N/A"}]

def extract_txt(file_stream, filename):
    content = file_stream.read().decode("utf-8", errors="ignore")
    return [{"text": content.strip(), "filename": filename, "page": "N/A"}]

def extract_md(file_stream, filename):
    return extract_txt(file_stream, filename)

def process_file(file_obj, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_pdf(file_obj, filename)
    elif ext == ".docx":
        return extract_docx(file_obj, filename)
    elif ext == ".txt":
        return extract_txt(file_obj, filename)
    elif ext == ".md":
        return extract_md(file_obj, filename)
    else:
        return []

# ==========================================
# 3. CHUNKING & EMBEDDING
# ==========================================

def chunk_text(pages_data, chunk_size=500, overlap=50):
    """Splits extracted text into overlapping chunks keeping metadata."""
    chunked_docs = []
    for data in pages_data:
        text = data["text"]
        words = text.split()
        if not words:
            continue
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            chunked_docs.append({
                "text": chunk_str,
                "filename": data["filename"],
                "page": data["page"]
            })
            if i + chunk_size >= len(words):
                break
            i += (chunk_size - overlap)
    return chunked_docs

def build_vector_store(chunks):
    if not chunks:
        return None, None
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    # Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return embeddings, index

# ==========================================
# 4. HYBRID SEARCH (FAISS + KEYWORD)
# ==========================================

def keyword_search(query, chunks):
    keywords = re.findall(r'\w+', query.lower())
    scores = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(kw) for kw in keywords)
        scores.append(score)
    return np.array(scores)

def hybrid_search(query, top_k=3):
    if not st.session_state.chunks or st.session_state.faiss_index is None:
        return []
    
    # 1. FAISS Semantic Search Score
    query_vector = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = st.session_state.faiss_index.search(query_vector, len(st.session_state.chunks))
    
    # Convert L2 distance to normalized similarity score [0, 1]
    raw_distances = distances[0]
    faiss_scores = 1 / (1 + raw_distances)
    
    # 2. Keyword Search Score
    kw_scores = keyword_search(query, st.session_state.chunks)
    if kw_scores.max() > 0:
        kw_scores = kw_scores / kw_scores.max()  # Normalize
        
    # Combine scores (50% Semantic + 50% Keyword)
    combined_scores = 0.5 * faiss_scores + 0.5 * kw_scores
    
    # Rank indices
    ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    results = [st.session_state.chunks[idx] for idx in ranked_indices]
    return results

# ==========================================
# 5. GOOGLE DRIVE LINK HANDLER
# ==========================================

def download_from_gdrive(url):
    temp_dir = tempfile.mkdtemp()
    try:
        if "folders" in url:
            files = gdown.download_folder(url, output=temp_dir, quiet=True)
            return files or []
        else:
            file_path = gdown.download(url, quiet=True, fuzzy=True)
            return [file_path] if file_path else []
    except Exception as e:
        st.error(f"Error fetching from Google Drive: {e}")
        return []

# ==========================================
# 6. STREAMLIT UI
# ==========================================

st.title("📄 AI Document Assistant")

# Sidebar - Document Ingestion
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    
    st.subheader("Or Google Drive Link")
    gdrive_link = st.text_input("Paste Drive File/Folder URL")
    
    if st.button("Process Documents"):
        all_raw_data = []
        
        # Process Local Uploads
        if uploaded_files:
            for file in uploaded_files:
                extracted = process_file(file, file.name)
                all_raw_data.extend(extracted)
                
        # Process Google Drive Links
        if gdrive_link:
            with st.spinner("Downloading files from Google Drive..."):
                downloaded_paths = download_from_gdrive(gdrive_link)
                for path in downloaded_paths:
                    if path and os.path.isfile(path):
                        fname = os.path.basename(path)
                        with open(path, "rb") as f:
                            extracted = process_file(f, fname)
                            all_raw_data.extend(extracted)
                            
        # Chunking & Embeddings
        if all_raw_data:
            chunks = chunk_text(all_raw_data)
            embeddings, index = build_vector_store(chunks)
            
            # Store in Session State (Embed once)
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.faiss_index = index
            
            st.success(f"Processing Complete! Created **{len(chunks)}** text chunks.")
        else:
            st.warning("No valid text extracted from the inputs.")

# Main Interface - Document Information & Question Answering
if st.session_state.chunks:
    st.info(f"📊 **System Ready**: {len(st.session_state.chunks)} chunks stored in memory.")
    
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        # Hybrid Search
        retrieved_chunks = hybrid_search(query, top_k=3)
        
        # Build Context
        context_str = "\n\n".join([
            f"[Source: {c['filename']}, Page: {c['page']}]\n{c['text']}" 
            for c in retrieved_chunks
        ])
        
        # Prompt Setup
        system_prompt = (
            "You are a helpful document assistant. Answer the user's question strictly using "
            "only the provided context below. If the information is not available in the context, "
            "explicitly state: 'The required information is not available in the provided documents.'"
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
        
        # Call Groq API securely
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
            client = Groq(api_key=groq_api_key)
            
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            answer = response.choices[0].message.content
            
            st.markdown("### Answer")
            st.write(answer)
            
            # Display Retrieved Sources below the answer
            st.markdown("---")
            st.markdown("### Retrieved Sources")
            for idx, src in enumerate(retrieved_chunks, 1):
                with st.expander(f"Source {idx}: {src['filename']} (Page: {src['page']})"):
                    st.write(src["text"])
                    
        except KeyError:
            st.error("`GROQ_API_KEY` not found in Streamlit Secrets. Please set it in `.streamlit/secrets.toml`.")
        except Exception as e:
            st.error(f"API Error: {e}")
else:
    st.info("Upload documents or paste a Google Drive link from the sidebar to get started.")
