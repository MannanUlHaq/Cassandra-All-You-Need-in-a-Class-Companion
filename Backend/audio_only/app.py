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

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Cassandra Audio-Only Backend is running"}), 200

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
    logger.info(f"Processing audio-only session for: {folder_path}")
    
    try:
        # 1. List files in the session folder
        files_response = supabase.storage.from_(BUCKET_AUDIO).list(folder_path)
        if not files_response:
            return jsonify({"success": False, "error": "No files found"}), 404

        # 2. Extract ONLY audio files formatted as audio_part_{timestamp}.webm
        audio_chunks = [
            f['name'] for f in files_response 
            if f['name'].startswith('audio_part_') and f['name'].endswith('.webm')
        ]
        
        if not audio_chunks:
            return jsonify({"success": False, "error": "No raw audio parts found"}), 404

        # Sort chronologically by the timestamp in the filename
        def extract_timestamp(filename):
            try:
                # audio_part_1760000000.webm -> 1760000000
                return int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                return 0

        audio_chunks.sort(key=extract_timestamp)
        
        # 3. Stitching Process
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Stitching {len(audio_chunks)} audio parts into one.")
            combined_audio = AudioSegment.empty()
            
            for chunk_name in audio_chunks:
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

            # 4. Export Final Audio
            final_filename = "final_audio.wav"
            if len(combined_audio) > 0:
                final_local_path = os.path.join(temp_dir, final_filename)
                combined_audio.export(final_local_path, format="wav")
                
                remote_final_path = f"{folder_path}/{final_filename}"
                with open(final_local_path, 'rb') as f:
                    supabase.storage.from_(BUCKET_AUDIO).upload(
                        path=remote_final_path,
                        file=f,
                        file_options={"upsert": "true", "content-type": "audio/wav"}
                    )
                
                # 5. Cleanup Step - Delete raw chunks
                paths_to_remove = [f"{folder_path}/{chunk}" for chunk in audio_chunks]
                supabase.storage.from_(BUCKET_AUDIO).remove(paths_to_remove)
                logger.info(f"Cleaned up {len(paths_to_remove)} raw audio chunks.")

                return jsonify({
                    "success": True, 
                    "message": f"Successfully stitched {len(audio_chunks)} parts into final_audio.wav",
                    "final_file": final_filename
                })
            else:
                return jsonify({"success": False, "error": "Combined audio resulted in an empty file."}), 500

    except Exception as e:
        logger.error(f"Error merging audio: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
