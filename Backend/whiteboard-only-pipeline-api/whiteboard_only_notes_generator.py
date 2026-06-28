import os
import argparse
import requests
import tempfile
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Note: Remember to update this URL to match your deployed Whiteboard Modal backend!
MODAL_WB_API_URL = "https://cassandra-classcompanion--whiteboard-notes-generator-fas-d872bd.modal.run/generate-whiteboard-notes"

class CloudWhiteboardNotesGenerator:
    def __init__(self, course_id):
        self.course_id = course_id
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.wb_json_str = None
        self.audio_json_str = None
        self.lecture_num = 1 # Default starting point

        self.md_path = os.path.join(self.temp_dir, "whiteboard_notes.md")
        self.html_path = os.path.join(self.temp_dir, "whiteboard_notes.html")

    def download_data(self):
        """Fetch whiteboard structured JSON and audio transcriptions from Supabase."""
        print(f"☁️ Downloading raw Whiteboard data for course: {self.course_id}")

        # 1. Download Whiteboard JSON
        try:
            wb_path = f"{self.course_id}_whiteboard_structured.json"
            print(f"   -> Looking for: extracted_whiteboard_content/{wb_path}")
            wb_bytes = self.supabase.storage.from_('extracted_whiteboard_content').download(wb_path)
            self.wb_json_str = wb_bytes.decode('utf-8')
            print("   ✅ Downloaded Structured Whiteboard Data.")
        except Exception as e:
            print(f"❌ Failed to find Whiteboard JSON at 'extracted_whiteboard_content/{wb_path}'")
            print(f"   Supabase Error: {e}")
            return False

        # 2. Download Audio Transcription
        try:
            audio_path = f"{self.course_id}/{self.course_id}_results.json"
            print(f"   -> Looking for: audio_transcription/{audio_path}")
            audio_bytes = self.supabase.storage.from_('audio_transcription').download(audio_path)
            self.audio_json_str = audio_bytes.decode('utf-8')
            print("   ✅ Downloaded Audio Transcription.")
        except Exception as e:
            print(f"❌ Failed to find Audio JSON at 'audio_transcription/{audio_path}'")
            print(f"   Supabase Error: {e}")
            return False

        return True

    def determine_lecture_number(self):
        """Looks at the generated_notes bucket to find the next available Lecture folder number."""
        print("🔍 Determining lecture folder number...")
        try:
            # Check for existing folders/files in the course_id directory
            existing_items = self.supabase.storage.from_('generated_notes').list(self.course_id)
            if existing_items:
                # Count items that start with "Lecture_" to find the next number
                lecture_folders = [f for f in existing_items if f.get('name', '').startswith('Lecture_')]
                self.lecture_num = len(lecture_folders) + 1
        except Exception as e:
            print(f"ℹ️ Note: Could not fetch existing folders. Starting at Lecture 1.")
        
        print(f"🎯 Assigned Folder: Lecture_{self.lecture_num}")

    def generate_notes_via_modal(self):
        """Send JSON strings to Modal, receive JSON back containing Markdown and HTML strings."""
        print("🚀 Sending data to Modal GPU Pipeline (Semantic Alignment -> RAG -> Summarize -> HTML Generation)...")

        try:
            response = requests.post(
                MODAL_WB_API_URL,
                data={
                    'wb_json': self.wb_json_str,
                    'audio_json': self.audio_json_str,
                    'supabase_url': SUPABASE_URL,
                    'supabase_key': SUPABASE_KEY,
                    'deepseek_api_key': DEEPSEEK_API_KEY,
                    'course_id': self.course_id
                },
                timeout=3600 # 1 hour timeout
            )
            response.raise_for_status()

            data = response.json()

            with open(self.md_path, 'w', encoding='utf-8') as f:
                f.write(data.get('markdown', ''))

            with open(self.html_path, 'w', encoding='utf-8') as f:
                f.write(data.get('html', ''))

            print(f"✅ Successfully generated and saved MD and HTML to {self.temp_dir}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Modal API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Details: {e.response.text}")
            return False

    def upload_notes(self):
        """Upload final HTML and MD to the isolated Lecture folder."""
        base_folder = f"{self.course_id}/Lecture_{self.lecture_num}"
        print(f"📤 Uploading Final Notes to Supabase ({base_folder}/)...")

        md_destination_path = f"{base_folder}/{self.course_id}_Whiteboard_Lecture_{self.lecture_num}.md"
        html_destination_path = f"{base_folder}/{self.course_id}_Whiteboard_Lecture_{self.lecture_num}.html"

        print(f"📄 Target MD name: {md_destination_path}")
        print(f"📄 Target HTML name: {html_destination_path}")

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

    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files.")

def main():
    parser = argparse.ArgumentParser(description="Generate Whiteboard Notes via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    args = parser.parse_args()

    course_id = args.course_id
    generator = CloudWhiteboardNotesGenerator(course_id)

    try:
        if not generator.download_data(): return
        
        # Figure out which folder to use before uploading
        generator.determine_lecture_number()
        
        if not generator.generate_notes_via_modal(): return
        
        generator.upload_notes()
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()
