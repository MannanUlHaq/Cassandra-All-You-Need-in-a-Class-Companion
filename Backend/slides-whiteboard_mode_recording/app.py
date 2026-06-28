import os
import tempfile
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydub import AudioSegment
from supabase import create_client, Client

# Logging for Heroku troubleshooting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes under /api/
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Credentials from Heroku Config Vars
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_AUDIO = "lecture_audio"
BUCKET_IMAGES = "lecture_images"

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Cassandra Hybrid (Slide-Centric) Backend is running"}), 200

@app.route('/api/start_session', methods=['POST'])
def start_session():
    return jsonify({"success": True})

@app.route('/api/end_session', methods=['POST'])
def end_session():
    data = request.get_json(silent=True)
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    session_id = data.get('session_id')
    course_id = data.get('course_id')
    
    if not session_id or not course_id:
        return jsonify({"success": False, "error": "Missing session_id or course_id"}), 400

    folder_path = f"{course_id}/{session_id}"
    logger.info(f"Processing hybrid audio for: {folder_path}")
    
    try:
        # 1. List all chunks in the session folder
        files_response = supabase.storage.from_(BUCKET_AUDIO).list(folder_path)
        if not files_response:
            return jsonify({"success": False, "error": "No files found"}), 404

        # 2. Extract ONLY audio files formatted as slide_{num}_{timestamp}.webm
        audio_chunks = [
            f['name'] for f in files_response 
            if f['name'].startswith('slide_') and f['name'].endswith('.webm')
        ]
        
        if not audio_chunks:
            return jsonify({"success": False, "error": "No raw audio chunks found"}), 404

        # Sort chronologically by timestamp
        def extract_timestamp(filename):
            try:
                # slide_1_1760000000.webm -> 1760000000
                return int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                return 0

        audio_chunks.sort(key=extract_timestamp)
        
        # 3. Group chunks strictly by their Slide Number
        slide_dict = {}
        for filename in audio_chunks:
            parts = filename.split('_')
            # Expected Format: ['slide', '1', '123456.webm']
            if len(parts) >= 3:
                slide_num = parts[1]
                slide_key = f"slide_{slide_num}"
                
                if slide_key not in slide_dict:
                    slide_dict[slide_key] = []
                slide_dict[slide_key].append(filename)

        # 4. Stitching Process
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for slide_key, chunks in slide_dict.items():
                logger.info(f"Stitching {len(chunks)} chunks for {slide_key}")
                combined_audio = AudioSegment.empty()
                
                for chunk_name in chunks:
                    remote_path = f"{folder_path}/{chunk_name}"
                    local_path = os.path.join(temp_dir, chunk_name)
                    
                    try:
                        chunk_data = supabase.storage.from_(BUCKET_AUDIO).download(remote_path)
                        if not chunk_data or len(chunk_data) == 0:
                            continue
                            
                        with open(local_path, 'wb') as f:
                            f.write(chunk_data)
                        
                        segment = AudioSegment.from_file(local_path, format="webm")
                        combined_audio += segment
                    except Exception as chunk_err:
                        logger.error(f"Error processing chunk {chunk_name}: {chunk_err}")

                # Export Final Audio per Slide
                if len(combined_audio) > 0:
                    final_filename = f"{slide_key}_final.wav"
                    final_local_path = os.path.join(temp_dir, final_filename)
                    combined_audio.export(final_local_path, format="wav")
                    
                    remote_final_path = f"{folder_path}/{final_filename}"
                    with open(final_local_path, 'rb') as f:
                        supabase.storage.from_(BUCKET_AUDIO).upload(
                            path=remote_final_path,
                            file=f,
                            file_options={"upsert": "true", "content-type": "audio/wav"}
                        )
                    results.append(final_filename)

        # 5. Cleanup Step - Delete raw chunks
        if results:
            paths_to_remove = [f"{folder_path}/{chunk}" for chunk in audio_chunks]
            supabase.storage.from_(BUCKET_AUDIO).remove(paths_to_remove)
            logger.info(f"Cleaned up {len(paths_to_remove)} raw hybrid chunks.")

        return jsonify({
            "success": True, 
            "message": f"Merged {len(results)} slides and cleaned up raw files.",
            "final_files": results
        })

    except Exception as e:
        logger.error(f"Error merging hybrid audio: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
