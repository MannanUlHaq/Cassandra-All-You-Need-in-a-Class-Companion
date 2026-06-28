import os
import tempfile
import logging
import re
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
    return jsonify({"status": "online", "message": "Cassandra Whiteboard Backend is running"}), 200

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
    logger.info(f"Processing whiteboard audio for: {folder_path}")
    
    try:
        # FIX 1: Increased limit to 1000 to prevent missing files during long lectures
        images_response = supabase.storage.from_(BUCKET_IMAGES).list(folder_path, {"limit": 1000})
        valid_board_nums = set()
        if images_response:
            for f in images_response:
                if f['name'].startswith('wb_') and f['name'].endswith('.jpg'):
                    match = re.match(r'wb_(\d+)\.jpg', f['name'])
                    if match:
                        valid_board_nums.add(int(match.group(1)))

        # FIX 1 (Continued): Increased limit for audio chunks
        files_response = supabase.storage.from_(BUCKET_AUDIO).list(folder_path, {"limit": 1000})
        if not files_response:
            return jsonify({"success": False, "error": "No files found"}), 404

        # Extract ONLY whiteboard audio files (raw chunks)
        audio_chunks = [
            f['name'] for f in files_response 
            if f['name'].startswith('wb_') and f['name'].endswith('.webm')
        ]
        
        if not audio_chunks:
            return jsonify({"success": False, "error": "No whiteboard raw chunks found"}), 404

        # Sort chronologically by the exact client timestamp embedded in the filename
        def extract_timestamp(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                return 0

        audio_chunks.sort(key=extract_timestamp)
        
        # Group chunks robustly using Regex strictly for 'wb_'
        boards_dict = {}
        for filename in audio_chunks:
            match = re.match(r'(wb_\d+)', filename)
            if match:
                board_key = match.group(1) 
            else:
                parts = filename.split('_')
                board_key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else filename.replace('.webm', '')

            if board_key not in boards_dict:
                boards_dict[board_key] = []
            boards_dict[board_key].append(filename)

        def get_numeric_index(key):
            try:
                return int(re.search(r'\d+', key).group())
            except (ValueError, AttributeError):
                return 0
                
        sorted_keys = sorted(boards_dict.keys(), key=get_numeric_index)

        # 3. Handle Orphaned Audio (Audio exists, but image was never captured)
        merged_boards_dict = {}
        for key in sorted_keys:
            board_num = get_numeric_index(key)
            target_num = board_num
            
            # If this board has no image AND it's not the only board
            if target_num not in valid_board_nums and len(valid_board_nums) > 0:
                smaller_valids = [n for n in valid_board_nums if n < board_num]
                if smaller_valids:
                    target_num = max(smaller_valids) # Map to the previous valid board
                    logger.info(f"Image for board {board_num} missing. Merging its audio into board {target_num}.")
            
            target_key = f"wb_{target_num}"
            if target_key not in merged_boards_dict:
                merged_boards_dict[target_key] = []
            merged_boards_dict[target_key].extend(boards_dict[key])

        sorted_merged_keys = sorted(merged_boards_dict.keys(), key=get_numeric_index)

        # 4. Stitching Process
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for board_key in sorted_merged_keys:
                chunks = merged_boards_dict[board_key]
                logger.info(f"Stitching {len(chunks)} chunks for {board_key}")
                
                # FIX 2: Initialize as None instead of AudioSegment.empty() to inherit correct sample rate/channels
                combined_audio = None 
                
                for chunk_name in chunks:
                    remote_path = f"{folder_path}/{chunk_name}"
                    local_path = os.path.join(temp_dir, chunk_name)
                    
                    try:
                        chunk_data = supabase.storage.from_(BUCKET_AUDIO).download(remote_path)
                        
                        # Prevent pydub crashes on empty chunks
                        if not chunk_data or len(chunk_data) == 0:
                            continue
                            
                        with open(local_path, 'wb') as f:
                            f.write(chunk_data)
                        
                        segment = AudioSegment.from_file(local_path, format="webm")
                        
                        # FIX 2 (Continued): Append properly based on the first chunk's metadata
                        if combined_audio is None:
                            combined_audio = segment
                        else:
                            combined_audio += segment
                            
                    except Exception as chunk_err:
                        logger.error(f"Error processing chunk {chunk_name}: {chunk_err}")

                # Export and Upload Final Version
                if combined_audio is not None:
                    final_filename = f"{board_key}_final.wav"
                    final_local_path = os.path.join(temp_dir, final_filename)
                    combined_audio.export(final_local_path, format="wav")
                    
                    remote_final_path = f"{folder_path}/{final_filename}"
                    
                    # FIX 3: Pass the file path string directly to Supabase instead of the open file object
                    supabase.storage.from_(BUCKET_AUDIO).upload(
                        path=remote_final_path,
                        file=final_local_path,
                        file_options={"upsert": "true", "content-type": "audio/wav"}
                    )
                    results.append(final_filename)

        # 5. Cleanup Step - Delete the raw .webm chunks
        if results:
            paths_to_remove = [f"{folder_path}/{chunk}" for chunk in audio_chunks]
            supabase.storage.from_(BUCKET_AUDIO).remove(paths_to_remove)
            logger.info(f"Cleaned up {len(paths_to_remove)} raw whiteboard chunks.")

        return jsonify({
            "success": True, 
            "message": f"Merged {len(results)} whiteboard tracks and cleaned up raw files.",
            "final_files": results
        })

    except Exception as e:
        logger.error(f"Error merging whiteboard audio: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
