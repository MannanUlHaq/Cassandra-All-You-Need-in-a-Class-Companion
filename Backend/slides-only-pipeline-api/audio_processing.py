import argparse
import os
import json
import tempfile
import requests
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ✅ Updated with your live Modal workspace URL
MODAL_API_URL = "https://cassandra-classcompanion--audio-ai-backend-fastapi-app.modal.run/process-audio"

class CloudAudioProcessor:
    def __init__(self, course_id, mode="slides_only"):
        self.course_id = course_id
        self.mode = mode
        self.results = {}
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.teacher_id = None
        self.teacher_sample_path = None
        self.audio_files_to_process = []

    def setup_cloud_data(self):
        """Fetches necessary IDs and downloads files from Supabase"""
        print(f"☁️ Setting up cloud data for course: {self.course_id}")

        response = self.supabase.table('class').select('teacher_id').eq('course_id', self.course_id).execute()
        if not response.data:
            print(f"❌ Could not find teacher_id for course_id {self.course_id}")
            return False

        self.teacher_id = response.data[0]['teacher_id']
        print(f"✅ Found teacher_id: {self.teacher_id}")

        # Download Teacher Voice Signature
        signature_file_path = None
        try:
            root_files = self.supabase.storage.from_('voice_signatures').list("")
            for f in root_files:
                if f['name'].startswith(self.teacher_id) and f['name'].endswith('.wav'):
                    signature_file_path = f['name']
                    break
        except Exception:
            pass

        if not signature_file_path:
            try:
                folder_files = self.supabase.storage.from_('voice_signatures').list(self.teacher_id)
                for f in folder_files:
                    if f['name'].endswith('.wav'):
                        signature_file_path = f"{self.teacher_id}/{f['name']}"
                        break
            except Exception:
                pass

        if not signature_file_path:
            print(f"❌ No .wav signature found for teacher {self.teacher_id}")
            return False

        sig_data = self.supabase.storage.from_('voice_signatures').download(signature_file_path)
        self.teacher_sample_path = os.path.join(self.temp_dir, f"teacher_{self.teacher_id}.wav")
        with open(self.teacher_sample_path, "wb") as f:
            f.write(sig_data)
        print("✅ Downloaded teacher voice signature")

        # Download Lecture Audio
        course_folders = self.supabase.storage.from_('lecture_audio').list(self.course_id)
        if not course_folders:
            print(f"❌ No audio folder found for course {self.course_id}")
            return False

        subfolder_name = course_folders[0]['name']
        audio_files = self.supabase.storage.from_('lecture_audio').list(f"{self.course_id}/{subfolder_name}")

        for file_meta in audio_files:
            if file_meta['name'].endswith('.wav') or file_meta['name'].endswith('.webm'): # Handled .webm compatibility if using browser defaults
                file_path = f"{self.course_id}/{subfolder_name}/{file_meta['name']}"
                audio_data = self.supabase.storage.from_('lecture_audio').download(file_path)
                local_path = os.path.join(self.temp_dir, file_meta['name'])
                with open(local_path, "wb") as f:
                    f.write(audio_data)
                self.audio_files_to_process.append(local_path)

        self.audio_files_to_process.sort()
        print(f"✅ Downloaded {len(self.audio_files_to_process)} lecture audio files")
        return True

    def transcribe_via_modal(self, audio_path):
        """Send audio to Modal GPU API for processing"""
        print(f"🚀 Sending {os.path.basename(audio_path)} to Modal GPU API...")
        try:
            with open(self.teacher_sample_path, 'rb') as teacher_file, \
                 open(audio_path, 'rb') as lecture_file:

                response = requests.post(
                    MODAL_API_URL,
                    files={
                        'teacher_audio': teacher_file,
                        'lecture_audio': lecture_file
                    },
                    timeout=900 # 15 minutes timeout for very large files
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ API error: {e}")
            return None

    def process_all_audio(self):
        """Process audio based on the selected mode"""
        if not self.audio_files_to_process:
            return False

        print(f"\n🎯 Processing in mode: {self.mode}")

        if self.mode == "audio_only":
            audio_file = self.audio_files_to_process[0]
            result = self.transcribe_via_modal(audio_file)
            if result:
                self.results["full_audio"] = result.get('teacher_text', "")
        else:
            prefix = "segment"
            if self.mode in ["slides_only", "slides_and_whiteboard"]:
                prefix = "slide"
            elif self.mode == "whiteboard_only":
                prefix = "whiteboard"

            for i, audio_file in enumerate(self.audio_files_to_process, 1):
                print(f"Processing {prefix} {i}...")
                result = self.transcribe_via_modal(audio_file)
                if result:
                    self.results[f"{prefix}_{i}"] = result.get('teacher_text', "")

        return True

    def clear_old_transcriptions(self):
        """Removes existing transcription files for this course to prevent stale data."""
        print(f"\n🧹 Clearing old transcriptions for course: {self.course_id}...")
        try:
            files = self.supabase.storage.from_('audio_transcription').list(self.course_id)
            if files:
                files_to_remove = []
                for f in files:
                    if f['name'] != '.emptyFolderPlaceholder':
                        files_to_remove.append(f"{self.course_id}/{f['name']}")
                
                if files_to_remove:
                    self.supabase.storage.from_('audio_transcription').remove(files_to_remove)
                    print(f"✅ Cleared {len(files_to_remove)} old transcription file(s).")
            else:
                print("✅ No old transcriptions found to clear.")
        except Exception as e:
            print(f"⚠️ Could not clear old transcriptions (might not exist yet): {e}")

    def save_and_upload_results(self):
        """Save results to JSON and upload to Supabase"""
        
        # 1. Clean old results before creating and uploading the new ones
        self.clear_old_transcriptions()

        # 2. Save new JSON
        local_json_path = os.path.join(self.temp_dir, f"{self.course_id}_results.json")

        with open(local_json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print("\n📤 Uploading results to Supabase...")
        with open(local_json_path, "rb") as f:
            destination_path = f"{self.course_id}/{self.course_id}_results.json"
            self.supabase.storage.from_('audio_transcription').upload(
                path=destination_path,
                file=f,
                file_options={"content-type": "application/json", "upsert": "true"}
            )

        print(f"✅ Results successfully uploaded to bucket: audio_transcription/{destination_path}")

    def cleanup(self):
        """Remove temporary directory and files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files")

def main():
    parser = argparse.ArgumentParser(description="Process Audio via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    parser.add_argument("--mode", default="slides_only", help="The processing mode")
    args = parser.parse_args()

    course_id = args.course_id
    mode = args.mode

    if MODAL_API_URL == "https://<YOUR-MODAL-WORKSPACE>--audio-ai-backend-fastapi-app.modal.run/process-audio":
        print("⚠️  PLEASE DEPLOY MODAL API FIRST AND UPDATE 'MODAL_API_URL' IN THIS SCRIPT!")
        return

    processor = CloudAudioProcessor(course_id=course_id, mode=mode)

    try:
        if not processor.setup_cloud_data():
            return
        processor.process_all_audio()
        processor.save_and_upload_results()
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
