import os
import re
import requests
import gc
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

# --- Memory Optimizations for Heroku Free/Hobby Dynos ---
torch.set_num_threads(1)

# ---------------------------
# CONFIGURATION
# ---------------------------
app = Flask(__name__)
CORS(app)

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Fail fast if credentials are not found in the environment
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing Supabase credentials! Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")

SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1"

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation,resolution=merge-duplicates"
}

# File upload settings
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Model settings (Lightweight model to prevent OOM)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cpu"

# Increased Chunk Sizes
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"🔧 Loading embedding model ({EMBEDDING_MODEL}) on {DEVICE}...")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
print("✅ Model loaded successfully")

# ---------------------------
# SUPABASE REST API HELPERS
# ---------------------------
def supabase_select(table: str, filters: Dict[str, Any] = None, select: str = "*"):
    url = f"{SUPABASE_REST_URL}/{table}?select={select}"
    if filters:
        for key, value in filters.items():
            url += f"&{key}=eq.{value}"
    
    response = requests.get(url, headers=SUPABASE_HEADERS)
    response.raise_for_status()
    return response.json()

def supabase_insert(table: str, data: Dict[str, Any]):
    url = f"{SUPABASE_REST_URL}/{table}"
    response = requests.post(url, headers=SUPABASE_HEADERS, json=data)
    response.raise_for_status()
    return response.json()

# ---------------------------
# TEXT PROCESSING FUNCTIONS
# ---------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\?\!\(\)\-\+\=\/\*]', '', text)
    return text.strip()

def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or len(text) < 50:
        return []
    
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_words = words[-overlap//10:] if overlap > 0 else []
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk and len(current_chunk) > 50:
        chunks.append(current_chunk.strip())
    
    return chunks

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# FILE EXTRACTION FUNCTIONS
# ---------------------------
def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    print(f"📖 Processing PDF: {file_path}")
    reader = PdfReader(file_path)
    documents = []
    total_pages = len(reader.pages)
    
    for page_num in range(total_pages):
        text = reader.pages[page_num].extract_text()
        if text and text.strip():
            chunks = smart_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_idx, chunk_text in enumerate(chunks):
                documents.append({
                    "text": chunk_text,
                    "page": page_num + 1,
                    "chunk_index": chunk_idx
                })
    return documents

def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    print(f"📖 Processing DOCX: {file_path}")
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs if para.text.strip()]
    
    combined_text = "\n".join(full_text)
    chunks = smart_chunk_text(combined_text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    documents = []
    for chunk_idx, chunk_text in enumerate(chunks):
        documents.append({
            "text": chunk_text,
            "page": None,
            "chunk_index": chunk_idx
        })
    return documents

# ---------------------------
# VECTOR GENERATION & STORAGE
# ---------------------------
def generate_and_store_vectors(material_id: int, filename: str, documents: List[Dict[str, Any]], course_id: str, teacher_id: str):
    print(f"📥 Generating embeddings for {len(documents)} chunks...")
    batch_size = 4 
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_texts = [doc["text"] for doc in batch_docs]
        
        with torch.no_grad():
            embeddings = embedder.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            embeddings_list = embeddings.numpy()
        
        for doc, embedding in zip(batch_docs, embeddings_list):
            metadata = {
                "filename": filename,
                "page": doc['page'],
                "chunk_index": doc['chunk_index'],
                "total_chunks": len(documents),
                "material_id": material_id
            }
            
            supabase_insert('course_material', {
                'content': doc['text'],
                'metadata': metadata,
                'embedding': embedding.tolist(),
                'course_id': course_id,
                'teacher_id': teacher_id
            })
            
        del embeddings, embeddings_list, batch_texts, batch_docs
        gc.collect()

# ---------------------------
# MATERIAL TRACKING 
# ---------------------------
def get_material_status(filename: str, course_id: str) -> bool:
    try:
        result = supabase_select('documentation', filters={'course_id': course_id, 'file_name': filename}, select='doc_id')
        return len(result) > 0 if result else False
    except Exception as e:
        print(f"Error checking material status: {e}")
        return False

def get_all_materials(course_id: str) -> List[Dict[str, Any]]:
    try:
        result = supabase_select('documentation', filters={'course_id': course_id}, select='doc_id,file_name,file_type')
        return result if result else []
    except Exception as e:
        print(f"❌ Error fetching materials: {e}")
        return []

# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        course_id = request.form.get('course_id')
        teacher_id = request.form.get('teacher_id')
        
        # Proper Error Handling: Ensure mapping exists
        if not course_id or course_id == 'undefined' or course_id == 'null':
            return jsonify({'error': 'Missing Course ID context.'}), 400
            
        if not teacher_id or teacher_id == 'undefined' or teacher_id == 'null':
            return jsonify({'error': 'Missing Teacher ID mapping. Authentication required.'}), 401
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or missing file type. Allowed: PDF, DOCX'}), 400
        
        material_id = int(datetime.now().timestamp() * 1000)
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{material_id}_{filename}")
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        # 1. Check if it already exists in documentation
        if get_material_status(filename, course_id):
            return jsonify({'error': 'File already exists in this course.'}), 409
        
        file.save(file_path)
        
        # 2. Extract & Chunk
        documents = extract_text_from_pdf(file_path) if file_ext == 'pdf' else extract_text_from_docx(file_path)
        
        if not documents:
            return jsonify({'error': 'No readable text content found in the file'}), 400
        
        # 3. Vectorize & Store in `course_material`
        generate_and_store_vectors(material_id, filename, documents, course_id, teacher_id)
        
        # 4. Insert into `documentation` table ONLY if vectors succeed
        supabase_insert('documentation', {
            'file_name': filename,
            'file_type': file_ext.upper(),
            'course_id': course_id
        })
        
        return jsonify({'success': True, 'filename': filename}), 200
        
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500
    finally:
        # Guarantee the file is removed from Heroku dyno memory regardless of success or crash
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        gc.collect()

@app.route('/api/files', methods=['GET'])
def get_files():
    try:
        course_id = request.args.get('course_id')
        if not course_id or course_id == 'undefined':
             return jsonify({'error': 'Missing Course ID'}), 400
             
        materials = get_all_materials(course_id)
        return jsonify({'files': materials}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------------
# HEROKU LAUNCHER
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
