import os
import time
import convertapi
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'converted_pdfs')
ALLOWED_EXTENSIONS = {'pptx'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# --- SECURE API KEY CONFIGURATION ---
# Fetch the API key securely from the environment
convertapi_key = os.environ.get('CONVERTAPI_KEY')
if not convertapi_key:
    raise ValueError("CRITICAL ERROR: CONVERTAPI_KEY environment variable is not set!")

convertapi.api_credentials = convertapi_key
# ------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_pptx_to_pdf(input_file_path, output_file_path):
    """
    Sends the PPTX file to ConvertAPI and saves the resulting PDF.
    """
    print(f"Sending {os.path.basename(input_file_path)} to ConvertAPI...")
    
    result = convertapi.convert('pdf', {
        'File': input_file_path
    }, from_format='pptx')
    
    # Save to the target directory
    out_dir = os.path.dirname(output_file_path)
    saved_files = result.save_files(out_dir)
    
    # ConvertAPI saves it with the original filename. Rename it to our timestamped filename.
    if saved_files and len(saved_files) > 0:
        saved_file_path = saved_files[0]
        if os.path.abspath(saved_file_path) != os.path.abspath(output_file_path):
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            os.rename(saved_file_path, output_file_path)
            
    print("Conversion successful!")
    return True

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only .pptx files are allowed'}), 400
            
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        base_name = os.path.splitext(filename)[0]
        
        input_filename = f"{timestamp}_{filename}"
        output_filename = f"{timestamp}_{base_name}.pdf"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 1. Save uploaded file locally temporarily
        file.save(input_path)
        
        # 2. Convert to PDF using the API
        convert_pptx_to_pdf(input_path, output_path)
        
        if not os.path.exists(output_path):
            return jsonify({'success': False, 'error': 'Conversion failed - file not created'}), 500
            
        # 3. Clean up original .pptx to save server space
        try: os.remove(input_path)
        except: pass
            
        return jsonify({
            'success': True,
            'pdf_filename': output_filename
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
        return send_file(file_path, mimetype='application/pdf', as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Cassandra Conversion API running via Slidize'})

if __name__ == '__main__':
    # Use dynamic port for Azure/Heroku
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
