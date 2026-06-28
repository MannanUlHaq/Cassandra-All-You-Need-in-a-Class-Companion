import argparse
import os
import requests
import tempfile
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Your DeepSeek API Key needed by Modal
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# The Modal API URL for Notes Generator
MODAL_NOTES_API_URL = "https://cassandra-classcompanion--notes-generator-backend-fastapi-app.modal.run/generate-notes"

class CloudNotesGenerator:
    def __init__(self, course_id):
        self.course_id = course_id
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.slides_json_str = None
        self.audio_json_str = None
        self.lecture_num = 1 # Default starting point

        # Paths for our new text-based output formats
        self.md_path = os.path.join(self.temp_dir, "notes.md")
        self.html_path = os.path.join(self.temp_dir, "notes.html")

    def download_data(self):
        """Fetch decomposed slides and audio transcriptions from Supabase."""
        print(f"☁️ Downloading raw data for course: {self.course_id}")

        try:
            # 1. Download Decomposed Slides
            slides_path = f"{self.course_id}/{self.course_id}_Slides_structured.json"
            slides_bytes = self.supabase.storage.from_('decomposed_slides').download(slides_path)
            self.slides_json_str = slides_bytes.decode('utf-8')
            print("✅ Downloaded Decomposed Slides.")

            # 2. Download Audio Transcription
            audio_path = f"{self.course_id}/{self.course_id}_results.json"
            audio_bytes = self.supabase.storage.from_('audio_transcription').download(audio_path)
            self.audio_json_str = audio_bytes.decode('utf-8')
            print("✅ Downloaded Audio Transcription.")

            return True
        except Exception as e:
            print(f"❌ Error downloading files from Supabase: {e}")
            print("Make sure BOTH the structured slides and audio transcriptions exist in their respective buckets.")
            return False

    def generate_notes_via_modal(self):
        """Send JSON strings to Modal, receive JSON back containing Markdown and HTML strings."""
        print("🚀 Sending data to Modal GPU Pipeline...")

        try:
            response = requests.post(
                MODAL_NOTES_API_URL,
                data={
                    'slides_json': self.slides_json_str,
                    'audio_json': self.audio_json_str,
                    'supabase_url': SUPABASE_URL,
                    'supabase_key': SUPABASE_KEY,
                    'deepseek_api_key': DEEPSEEK_API_KEY,
                    'course_id': self.course_id
                },
                timeout=3600 # 1 hour timeout for LLM generation
            )
            response.raise_for_status()

            # The Modal API now returns a JSON object with 'markdown' and 'html'
            data = response.json()

            # Save Markdown locally
            with open(self.md_path, 'w', encoding='utf-8') as f:
                f.write(data.get('markdown', ''))

            # Save HTML locally
            with open(self.html_path, 'w', encoding='utf-8') as f:
                f.write(data.get('html', ''))

            print(f"✅ Successfully generated and saved MD and HTML to {self.temp_dir}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Modal API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Details: {e.response.text}")
            return False

    def determine_lecture_number(self):
        """Looks at the generated_notes bucket to find the next available Lecture number."""
        print("🔍 Determining lecture folder number...")
        try:
            # Check for existing folders in the course_id directory
            existing_items = self.supabase.storage.from_('generated_notes').list(self.course_id)
            if existing_items:
                # Count items that start with "Lecture_" to find the next number
                lecture_folders = [f for f in existing_items if f.get('name', '').startswith('Lecture_')]
                self.lecture_num = len(lecture_folders) + 1
        except Exception as e:
            print(f"ℹ️ Note: Could not fetch existing folders. Starting at Lecture 1.")
        
        print(f"🎯 Assigned Folder: Lecture_{self.lecture_num}")

    def upload_notes(self):
        """Upload final HTML and MD to the isolated Lecture folder."""
        base_folder = f"{self.course_id}/Lecture_{self.lecture_num}"
        print(f"📤 Uploading Final Notes to Supabase ({base_folder}/)...")

        md_destination_path = f"{base_folder}/{self.course_id}_Lecture_{self.lecture_num}.md"
        html_destination_path = f"{base_folder}/{self.course_id}_Lecture_{self.lecture_num}.html"

        # 1. Upload MD
        try:
            with open(self.md_path, "rb") as f:
                self.supabase.storage.from_('generated_notes').upload(
                    path=md_destination_path,
                    file=f,
                    file_options={"content-type": "text/markdown", "upsert": "true"}
                )
            print(f"  ⬆️ Saved MD to {md_destination_path}")
        except Exception as e:
            print(f"❌ MD Upload Failed: {e}")

        # 2. Upload HTML
        try:
            with open(self.html_path, "rb") as f:
                self.supabase.storage.from_('generated_notes').upload(
                    path=html_destination_path,
                    file=f,
                    file_options={"content-type": "text/html", "upsert": "true"}
                )
            print(f"  ⬆️ Saved HTML to {html_destination_path}")
        except Exception as e:
            print(f"❌ HTML Upload Failed: {e}")

    def copy_media_folder(self):
        """Copies the media folder so it lives right next to the HTML file."""
        dest_prefix = f"{self.course_id}/Lecture_{self.lecture_num}/media"
        print(f"📁 Copying media files to {dest_prefix}/...")
        
        media_prefix = f"{self.course_id}/media"
        
        try:
            # List all files inside the media folder in the decomposed_slides bucket
            media_files = self.supabase.storage.from_('decomposed_slides').list(media_prefix)
            
            if not media_files:
                print("ℹ️ No media files found to copy.")
                return

            copied_count = 0
            for file_meta in media_files:
                file_name = file_meta.get('name')
                
                # Skip empty folder placeholders or hidden files
                if not file_name or file_name == ".emptyFolderPlaceholder":
                    continue

                source_path = f"{media_prefix}/{file_name}"
                dest_path = f"{dest_prefix}/{file_name}"
                
                # Download from source bucket
                file_bytes = self.supabase.storage.from_('decomposed_slides').download(source_path)
                
                # Determine basic content type
                content_type = "image/png" if file_name.endswith(".png") else "application/octet-stream"
                if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
                    content_type = "image/jpeg"

                # Upload to the isolated lecture folder
                self.supabase.storage.from_('generated_notes').upload(
                    path=dest_path,
                    file=file_bytes,
                    file_options={"content-type": content_type, "upsert": "true"}
                )
                copied_count += 1
                
            print(f"✅ Successfully copied {copied_count} media files to {dest_prefix}/")
            
        except Exception as e:
            print(f"❌ Failed to copy media folder: {e}")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files.")

def main():
    parser = argparse.ArgumentParser(description="Generate Notes via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    args = parser.parse_args()

    course_id = args.course_id
    generator = CloudNotesGenerator(course_id)

    try:
        if not generator.download_data(): return
        
        # Figure out which folder to use before generating/uploading
        generator.determine_lecture_number()
        
        if not generator.generate_notes_via_modal(): return
        
        generator.upload_notes()
        generator.copy_media_folder()
        
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()
