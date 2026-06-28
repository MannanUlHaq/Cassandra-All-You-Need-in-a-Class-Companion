import argparse
import os
import requests
import tempfile
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Note: Remember to update this URL to match your deployed Combined Modal backend!
MODAL_COMBINED_API_URL = "https://cassandra-classcompanion--combined-notes-generator-fastapi-app.modal.run/generate-combined-notes"

class CloudCombinedNotesGenerator:
    def __init__(self, course_id):
        self.course_id = course_id
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.lecture_num = 1

        self.slides_json_str = None
        self.wb_json_str = None
        self.audio_json_str = None

        self.md_path = os.path.join(self.temp_dir, "combined_notes.md")
        self.html_path = os.path.join(self.temp_dir, "combined_notes.html")

    def download_data(self):
        """Fetch slides, whiteboard JSON, and audio transcriptions from Supabase."""
        print(f"☁️ Downloading raw data for course: {self.course_id}")

        # 1. Download Slides
        try:
            slides_path = f"{self.course_id}/{self.course_id}_Slides_structured.json"
            print(f"   -> Looking for: decomposed_slides/{slides_path}")
            slides_bytes = self.supabase.storage.from_('decomposed_slides').download(slides_path)
            self.slides_json_str = slides_bytes.decode('utf-8')
            print("   ✅ Downloaded Structured Slides.")
        except Exception as e:
            print(f"❌ Failed to find Slides JSON at 'decomposed_slides/{slides_path}'. Error: {e}")
            return False

        # 2. Download Whiteboard JSON
        try:
            wb_path = f"{self.course_id}_whiteboard_structured.json"
            print(f"   -> Looking for: extracted_whiteboard_content/{wb_path}")
            wb_bytes = self.supabase.storage.from_('extracted_whiteboard_content').download(wb_path)
            self.wb_json_str = wb_bytes.decode('utf-8')
            print("   ✅ Downloaded Structured Whiteboard Data.")
        except Exception as e:
            print(f"❌ Failed to find Whiteboard JSON at 'extracted_whiteboard_content/{wb_path}'. Error: {e}")
            return False

        # 3. Download Audio Transcription
        try:
            audio_path = f"{self.course_id}/{self.course_id}_results.json"
            print(f"   -> Looking for: audio_transcription/{audio_path}")
            audio_bytes = self.supabase.storage.from_('audio_transcription').download(audio_path)
            self.audio_json_str = audio_bytes.decode('utf-8')
            print("   ✅ Downloaded Audio Transcription.")
        except Exception as e:
            print(f"❌ Failed to find Audio JSON at 'audio_transcription/{audio_path}'. Error: {e}")
            return False

        return True

    def determine_lecture_number(self):
        """Looks at the generated_notes bucket to find the next available Lecture number."""
        print("🔍 Determining lecture folder number...")
        try:
            existing_items = self.supabase.storage.from_('generated_notes').list(self.course_id)
            if existing_items:
                lecture_folders = [f for f in existing_items if f.get('name', '').startswith('Lecture_')]
                self.lecture_num = len(lecture_folders) + 1
        except Exception as e:
            print(f"ℹ️ Note: Could not fetch existing folders. Starting at Lecture 1.")
        
        print(f"🎯 Assigned Folder: Lecture_{self.lecture_num}")

    def generate_notes_via_modal(self):
        print("🚀 Sending data to Modal GPU Pipeline (Parallel Explain -> Interleave Merge -> Summarize -> HTML Generation)...")

        try:
            response = requests.post(
                MODAL_COMBINED_API_URL,
                data={
                    'slides_json': self.slides_json_str,
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

            print(f"✅ Successfully generated and saved MD and HTML locally to {self.temp_dir}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Modal API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Details: {e.response.text}")
            return False

    def upload_notes(self):
        """Upload final HTML and MD to the isolated Lecture folder."""
        base_folder = f"{self.course_id}/Lecture_{self.lecture_num}"
        print(f"📤 Uploading Final Combined Notes to Supabase ({base_folder}/)...")

        md_destination_path = f"{base_folder}/{self.course_id}_Combined_Lecture_{self.lecture_num}.md"
        html_destination_path = f"{base_folder}/{self.course_id}_Combined_Lecture_{self.lecture_num}.html"

        try:
            with open(self.md_path, "rb") as f:
                self.supabase.storage.from_('generated_notes').upload(
                    path=md_destination_path, file=f, file_options={"content-type": "text/markdown", "upsert": "true"}
                )
            print(f"✅ Upload Complete! Saved to generated_notes/{md_destination_path}")
        except Exception as e:
            print(f"❌ MD Upload Failed: {e}")

        try:
            with open(self.html_path, "rb") as f:
                self.supabase.storage.from_('generated_notes').upload(
                    path=html_destination_path, file=f, file_options={"content-type": "text/html", "upsert": "true"}
                )
            print(f"✅ Upload Complete! Saved to generated_notes/{html_destination_path}")
        except Exception as e:
            print(f"❌ HTML Upload Failed: {e}")

    def copy_media_folder(self):
        """Copies the slide media folder so it lives right next to the HTML file."""
        dest_prefix = f"{self.course_id}/Lecture_{self.lecture_num}/media"
        print(f"📁 Copying slide media files to {dest_prefix}/...")
        media_prefix = f"{self.course_id}/media"
        
        try:
            media_files = self.supabase.storage.from_('decomposed_slides').list(media_prefix)
            if not media_files:
                print("ℹ️ No media files found to copy.")
                return

            copied_count = 0
            for file_meta in media_files:
                file_name = file_meta.get('name')
                if not file_name or file_name == ".emptyFolderPlaceholder":
                    continue

                source_path = f"{media_prefix}/{file_name}"
                dest_path = f"{dest_prefix}/{file_name}"
                
                file_bytes = self.supabase.storage.from_('decomposed_slides').download(source_path)
                
                content_type = "image/png" if file_name.endswith(".png") else "application/octet-stream"
                if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
                    content_type = "image/jpeg"

                self.supabase.storage.from_('generated_notes').upload(
                    path=dest_path, file=file_bytes, file_options={"content-type": content_type, "upsert": "true"}
                )
                copied_count += 1
                
            print(f"✅ Successfully copied {copied_count} slide media files to {dest_prefix}/")
        except Exception as e:
            print(f"❌ Failed to copy media folder: {e}")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files.")

def main():
    parser = argparse.ArgumentParser(description="Generate Hybrid Notes via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    args = parser.parse_args()

    course_id = args.course_id
    generator = CloudCombinedNotesGenerator(course_id)

    try:
        if not generator.download_data(): return
        
        generator.determine_lecture_number()
        
        if not generator.generate_notes_via_modal(): return
        
        generator.upload_notes()
        generator.copy_media_folder()
        
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()