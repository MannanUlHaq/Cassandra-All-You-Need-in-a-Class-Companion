import argparse
import os
import requests
import json
import tempfile
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Note: Remember to update this URL to match your deployed Audio Modal backend!
MODAL_AUDIO_API_URL = "https://cassandra-classcompanion--audio-notes-generator-fastapi-app.modal.run/generate-audio-notes"

class CloudAudioNotesGenerator:
    def __init__(self, course_id):
        self.course_id = course_id
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.lecture_num = 1

        self.agenda_text = None
        self.transcript_text = None

        self.md_path = os.path.join(self.temp_dir, "audio_notes.md")
        self.html_path = os.path.join(self.temp_dir, "audio_notes.html")

    def download_data(self):
        """Fetch agenda and audio transcriptions from Supabase."""
        print(f"☁️ Downloading raw data for course: {self.course_id}")

        # 1. Download Agenda (From lecture_agenda bucket, handling nested folders)
        try:
            print(f"   -> Searching for Agenda in bucket 'lecture_agenda' under '{self.course_id}/'...")

            items = self.supabase.storage.from_('lecture_agenda').list(self.course_id)
            agenda_path = None

            if isinstance(items, list):
                for item in items:
                    if item['name'].endswith('.txt') and 'agenda' in item['name'].lower():
                        agenda_path = f"{self.course_id}/{item['name']}"
                        break

                    if '.' not in item['name'] and item['name'] != '.emptyFolderPlaceholder':
                        sub_path = f"{self.course_id}/{item['name']}"
                        sub_items = self.supabase.storage.from_('lecture_agenda').list(sub_path)

                        if isinstance(sub_items, list):
                            for sub_item in sub_items:
                                if sub_item['name'].endswith('.txt') and 'agenda' in sub_item['name'].lower():
                                    agenda_path = f"{sub_path}/{sub_item['name']}"
                                    break
                        if agenda_path:
                            break

            if not agenda_path:
                print(f"❌ Failed to find any agenda file inside 'lecture_agenda/{self.course_id}/'")
                return False

            print(f"   -> Found Agenda at: lecture_agenda/{agenda_path}")
            agenda_bytes = self.supabase.storage.from_('lecture_agenda').download(agenda_path)
            self.agenda_text = agenda_bytes.decode('utf-8')
            print("   ✅ Downloaded Agenda Text.")

        except Exception as e:
            print(f"❌ Error fetching Agenda from 'lecture_agenda' bucket. Error: {e}")
            return False

        # 2. Download Audio Transcription
        try:
            audio_path = f"{self.course_id}/{self.course_id}_results.json"
            print(f"   -> Looking for: audio_transcription/{audio_path}")
            audio_bytes = self.supabase.storage.from_('audio_transcription').download(audio_path)

            audio_data = json.loads(audio_bytes.decode('utf-8'))
            if isinstance(audio_data, dict):
                self.transcript_text = " ".join([str(v) for v in audio_data.values() if isinstance(v, str)])
            elif isinstance(audio_data, list):
                self.transcript_text = " ".join([str(v) for v in audio_data if isinstance(v, str)])
            else:
                self.transcript_text = str(audio_data)

            print("   ✅ Downloaded Audio Transcription.")
        except Exception as e:
            print(f"❌ Failed to find Audio JSON. Error: {e}")
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
        print("🚀 Sending data to Modal GPU Pipeline (Audio Alignment -> RAG Explanations -> DeepSeek Cleaning -> HTML Generation)...")

        try:
            response = requests.post(
                MODAL_AUDIO_API_URL,
                data={
                    'agenda_text': self.agenda_text,
                    'audio_transcript': self.transcript_text,
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
        print(f"📤 Uploading Final Audio Notes to Supabase ({base_folder}/)...")

        md_destination_path = f"{base_folder}/{self.course_id}_Audio_Lecture_{self.lecture_num}.md"
        html_destination_path = f"{base_folder}/{self.course_id}_Audio_Lecture_{self.lecture_num}.html"

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

    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files.")

def main():
    parser = argparse.ArgumentParser(description="Generate Audio Notes via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    args = parser.parse_args()

    course_id = args.course_id
    generator = CloudAudioNotesGenerator(course_id)

    try:
        if not generator.download_data(): return
        generator.determine_lecture_number()
        if not generator.generate_notes_via_modal(): return
        generator.upload_notes()
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()