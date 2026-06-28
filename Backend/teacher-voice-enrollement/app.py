import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from supabase import create_client, Client

# ---------------------------
# CONFIGURATION
# ---------------------------
app = Flask(__name__)

# Enable CORS for all domains so your frontend can communicate with it
CORS(app)

# --- Supabase Configuration ---
# IMPORTANT: For backend operations like Storage, it is highly recommended to use 
# your SERVICE ROLE KEY, not the anon key. Find it in Supabase > Settings > API.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lchcfsgexlkjxrhfjrna.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "YOUR_SERVICE_ROLE_KEY_HERE")
BUCKET_NAME = "voice_signatures"

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Storage location for temporary local processing
UPLOAD_FOLDER = "temp_voice_signatures"
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm', 'ogg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify backend is running."""
    return jsonify({
        'status': 'healthy',
        'service': 'Cassandra Voice Enrollment API',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/voice-enroll', methods=['POST'])
def handle_voice_enrollment():
    """Endpoint triggered after frontend registration to receive audio file."""
    try:
        # 1. Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part found in the request'}), 400
        
        file = request.files['file']
        teacher_id = request.form.get('teacher_id', 'unknown_teacher')

        if file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # 2. Process and Upload to Supabase
        if file:
            # Generate a unique filename using the teacher's ID
            timestamp = int(datetime.now().timestamp())
            safe_filename = secure_filename(f"{teacher_id}_{timestamp}.wav")
            local_file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
            
            # Save locally first (temporarily)
            file.save(local_file_path)
            
            # Read the saved file and upload to Supabase Storage
            with open(local_file_path, 'rb') as f:
                res = supabase.storage.from_(BUCKET_NAME).upload(
                    file=f,
                    path=safe_filename,
                    file_options={"content-type": "audio/wav"}
                )
            
            print(f"✅ Successfully uploaded voice signature to Supabase for Teacher: {teacher_id}")
            
            # (Optional) Delete the local file to save server space after successful cloud upload
            os.remove(local_file_path)
            
            # Get the public URL (will work if your bucket is set to Public)
            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(safe_filename)
            
            return jsonify({
                'success': True,
                'message': 'Voice signature enrolled and saved to Supabase successfully',
                'teacher_id': teacher_id,
                'supabase_path': safe_filename,
                'url': public_url
            }), 200

    except Exception as e:
        print(f"❌ Error during voice upload: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == '__main__':
    print("🚀 Starting Cassandra Voice Enrollment Backend...")
    # Running on 0.0.0.0 exposes it to your local network/Docker
    app.run(host='0.0.0.0', port=5000, debug=True)
