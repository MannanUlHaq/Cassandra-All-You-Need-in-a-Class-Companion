import os
import json
import requests
import traceback
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import torch

# ---------------------------
# CONFIGURATION (AZURE ENV VARS)
# ---------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Ensure frontend can talk to backend regardless of domain

# Retrieve secrets from Azure App Service Environment Variables (Application Settings)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Validate critical environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ WARNING: SUPABASE_URL or SUPABASE_KEY environment variables are missing!")

SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ""

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
print(f"🔧 Loading embedding model on {DEVICE}...")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
print("✅ Embedding model loaded")
print("🤖 DeepSeek API configured (Targeting: deepseek-chat)")

# ---------------------------
# SUPABASE HELPERS
# ---------------------------
def require_supabase():
    """Ensure Supabase URL is configured before making requests to avoid Schema Errors"""
    if not SUPABASE_REST_URL:
        raise ValueError("SUPABASE_URL environment variable is missing in Azure. Cannot connect to database.")

def supabase_rpc(function_name: str, params: Dict[str, Any]):
    """Call Supabase RPC function with error handling"""
    require_supabase()
    url = f"{SUPABASE_REST_URL}/rpc/{function_name}"
    response = requests.post(url, headers=SUPABASE_HEADERS, json=params)
    
    if response.status_code != 200:
        error_detail = response.json() if response.text else {}
        raise Exception(f"Supabase RPC error: {error_detail.get('message', response.text)}")
    
    return response.json()

def get_chat_history(teacher_id: str, course_id: str, limit: int = 5):
    """Get last N messages from chat history for a specific course"""
    require_supabase()
    url = f"{SUPABASE_REST_URL}/teacher_chat_history"
    params = {
        "teacher_id": f"eq.{teacher_id}",
        "course_id": f"eq.{course_id}",
        "select": "message,response,timestamp",
        "order": "timestamp.desc",
        "limit": str(limit)
    }
    
    response = requests.get(url, headers=SUPABASE_HEADERS, params=params)
    response.raise_for_status()
    data = response.json()
    return list(reversed(data))

def save_chat_message(teacher_id: str, course_id: str, message: str, response: str):
    """Save chat message and response for a specific course"""
    require_supabase()
    import uuid
    url = f"{SUPABASE_REST_URL}/teacher_chat_history"
    data = {
        "teacher_chat_id": str(uuid.uuid4()),  
        "teacher_id": teacher_id,
        "course_id": course_id,
        "message": message,
        "response": response
    }
    
    resp = requests.post(url, headers=SUPABASE_HEADERS, json=data)
    if resp.status_code not in (200, 201):
        raise Exception(f"Failed to save chat: {resp.text}")
    return resp.json()

# ---------------------------
# RAG SEARCH - WITH FALLBACK
# ---------------------------
def search_similar_content(query: str, course_id: str, top_k: int = 4):
    """Search with automatic fallback to exact string matching"""
    try:
        with torch.no_grad():
            query_embedding = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
            if DEVICE == "cuda": query_embedding = query_embedding.cpu()
            query_embedding_list = query_embedding.numpy()[0].tolist()
        
        results = supabase_rpc("search_similar_chunks", {
            "query_embedding": query_embedding_list,
            "match_course_id": course_id,
            "match_count": top_k,
            "similarity_threshold": 0.3
        })
        return results
    except Exception as rpc_error:
        print(f"⚠️ RPC failed: {rpc_error}")
        return []

# ---------------------------
# LLM ANSWER GENERATION
# ---------------------------
SYSTEM_PROMPT = """You are an AI teaching assistant Cassandra. You help teachers with course-related questions based on textbook content and conversation history.

**Guidelines:**
- Provide clear, helpful answers based on the provided context
- Reference the conversation history when relevant
- Use simple, direct language
- If the context doesn't contain the answer, say "Based on the course materials, this information isn't covered."
- Provide direct answers without unnecessary preamble
- Think step-by-step to ensure accuracy"""

def build_context_with_history(retrieved_docs: List[Dict], chat_history: List[Dict]) -> str:
    context_parts = []
    
    if chat_history:
        context_parts.append("**Recent Conversation:**")
        for entry in chat_history:
            context_parts.append(f"Teacher: {entry['message']}")
            context_parts.append(f"Assistant: {entry['response']}")
        context_parts.append("")
    
    if retrieved_docs:
        context_parts.append("**Relevant Course Material:**")
        for i, doc in enumerate(retrieved_docs[:3], 1): 
            context_parts.append(f"{i}. {doc['content']}")
        context_parts.append("")
        
    return "\n".join(context_parts)

def generate_answer(question: str, context: str) -> str:
    user_prompt = f"{context}\n\n**Teacher's Question:** {question}\n\n**Assistant's Answer:**"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",  # Maps to DeepSeek-V3
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.4,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            return "I couldn't generate an answer at the moment. Please try again."
            
    except Exception as e:
        print(f"❌ Error generating answer via DeepSeek API: {e}")
        return "I encountered an error while processing your question. Please try again."

# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'device': DEVICE, 'llm': 'DeepSeek API'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # silent=True ensures it returns None instead of crashing if Content-Type is missing
        data = request.get_json(silent=True)
        
        if not data:
            return jsonify({'error': 'Invalid or missing JSON body. Ensure the frontend sends Content-Type: application/json'}), 400
            
        message = data.get('message', '').strip()
        teacher_id = data.get('teacher_id')
        course_id = data.get('course_id')
        
        if not message or not teacher_id or not course_id:
            return jsonify({'error': 'Message, teacher_id, and course_id are required fields'}), 400
        
        chat_history = get_chat_history(teacher_id, course_id, limit=5)
        search_results = search_similar_content(message, course_id, top_k=4)
        context = build_context_with_history(search_results, chat_history)
        
        answer = generate_answer(message, context)
        save_chat_message(teacher_id, course_id, message, answer)
        
        return jsonify({
            'response': answer,
            'sources': [{'content': doc['content'][:200] + '...'} for doc in search_results[:2]]
        }), 200
        
    except Exception as e:
        # Print the exact error line to Azure App Service logs
        print("\n" + "="*50)
        print("🔥 ERROR IN /api/chat ENDPOINT:")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        teacher_id = request.args.get('teacher_id')
        course_id = request.args.get('course_id')
        if not teacher_id or not course_id:
            return jsonify({'error': 'teacher_id and course_id required'}), 400
        limit = int(request.args.get('limit', 20))
        history = get_chat_history(teacher_id, course_id, limit)
        return jsonify({'history': history}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        data = request.get_json(silent=True)
        if not data:
             return jsonify({'error': 'Invalid JSON body'}), 400
             
        teacher_id = data.get('teacher_id')
        course_id = data.get('course_id')
        
        if not teacher_id or not course_id:
             return jsonify({'error': 'teacher_id and course_id required'}), 400
        
        require_supabase()
        url = f"{SUPABASE_REST_URL}/teacher_chat_history"
        params = {
            "teacher_id": f"eq.{teacher_id}",
            "course_id": f"eq.{course_id}"
        }
        
        response = requests.delete(url, headers=SUPABASE_HEADERS, params=params)
        response.raise_for_status()
        
        return jsonify({'success': True, 'message': 'Chat history cleared for course'}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ---------------------------
# AZURE DEPLOYMENT MAIN
# ---------------------------
if __name__ == '__main__':
    # Azure dynamically provides the port in an environment variable (WEBSITES_PORT or PORT).
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
