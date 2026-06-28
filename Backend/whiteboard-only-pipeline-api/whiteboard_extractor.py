import os
import json
import re
import argparse
import tempfile
import requests
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ⚠️ UPDATE THIS URL AFTER DEPLOYING TO MODAL
MODAL_WHITEBOARD_API_URL = "https://cassandra-classcompanion--whiteboard-ai-backend-fastapi-app.modal.run/process-whiteboards"

def natural_sort_key(s):
    """Sort files logically (wb_1, wb_2, wb_10) instead of alphabetically (wb_1, wb_10, wb_2)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class CloudWhiteboardProcessor:
    def __init__(self, course_id):
        self.course_id = course_id
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.local_image_paths = []
        self.output_json_path = None

    def download_whiteboards(self):
        """Downloads all whiteboard images from Supabase"""
        print(f"☁️ Searching for whiteboard images for course: {self.course_id}")

        # 1. Find the subfolder inside the course_id folder
        try:
            folders = self.supabase.storage.from_('lecture_images').list(self.course_id)
            if not folders:
                print(f"❌ No folders found for course {self.course_id} in lecture_images.")
                return False

            subfolder_name = folders[0]['name']
            folder_path = f"{self.course_id}/{subfolder_name}"

            # 2. List all images
            images = self.supabase.storage.from_('lecture_images').list(folder_path)
            image_names = [img['name'] for img in images if img['name'].lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_names:
                print(f"❌ No JPG/PNG images found in {folder_path}")
                return False

            # Sort them naturally (wb_1, wb_2... wb_10)
            image_names.sort(key=natural_sort_key)

            # 3. Download them
            print(f"📥 Found {len(image_names)} images. Downloading...")
            for img_name in image_names:
                file_path = f"{folder_path}/{img_name}"
                img_data = self.supabase.storage.from_('lecture_images').download(file_path)

                local_path = os.path.join(self.temp_dir, img_name)
                with open(local_path, "wb") as f:
                    f.write(img_data)
                self.local_image_paths.append(local_path)

            print("✅ All whiteboard images downloaded successfully.")
            return True

        except Exception as e:
            print(f"❌ Supabase Error: {e}")
            return False

    def process_via_modal(self):
        """Send all images as a batch request to Modal API"""
        print("🚀 Sending images to Modal GPU API for Qwen Extraction...")

        files_payload = []
        file_handles = [] # Keep track so we can safely close them later

        try:
            # Prepare the multipart form data
            for img_path in self.local_image_paths:
                f = open(img_path, 'rb')
                file_handles.append(f)
                filename = os.path.basename(img_path)
                # Ensure the field name 'images' perfectly matches the FastAPI parameter
                files_payload.append(('images', (filename, f, 'image/jpeg')))

            response = requests.post(
                MODAL_WHITEBOARD_API_URL,
                files=files_payload,
                timeout=3600 # 1 hour timeout for batch processing
            )
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "success":
                self.output_json_path = os.path.join(self.temp_dir, f"{self.course_id}_whiteboard_structured.json")
                with open(self.output_json_path, "w", encoding="utf-8") as out_f:
                    json.dump(data.get("results", []), out_f, indent=2, ensure_ascii=False)

                print("✅ Whiteboards completely processed and JSON generated!")
                return True
            else:
                print(f"❌ Modal processing error: {data.get('error')}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            if 'response' in locals() and response is not None:
                print(f"Server response: {response.text}")
            return False
        finally:
            # Close all opened file handles
            for f in file_handles:
                f.close()

    def clear_old_extracted_data(self):
        """Removes existing processed whiteboard JSON for this course to prevent stale data."""
        print(f"\n🧹 Clearing old extracted whiteboard data for course: {self.course_id}...")
        try:
            file_to_remove = f"{self.course_id}_whiteboard_structured.json"
            # Attempt to remove the old JSON file
            self.supabase.storage.from_('extracted_whiteboard_content').remove([file_to_remove])
            print("✅ Cleared old whiteboard JSON (if it existed).")
        except Exception as e:
            print(f"⚠️ Could not clear old whiteboard data: {e}")

    def upload_results(self):
        """Upload the structured JSON to extracted_whiteboard_content bucket"""
        if not self.output_json_path or not os.path.exists(self.output_json_path):
            return

        # 1. Clear out old JSON first
        self.clear_old_extracted_data()

        print("\n📤 Uploading structured JSON to Supabase...")
        try:
            with open(self.output_json_path, "rb") as f:
                destination_path = f"{self.course_id}_whiteboard_structured.json"
                self.supabase.storage.from_('extracted_whiteboard_content').upload(
                    path=destination_path,
                    file=f,
                    file_options={"content-type": "application/json", "upsert": "true"}
                )
            print(f"✅ Results successfully uploaded: extracted_whiteboard_content/{destination_path}")
        except Exception as e:
            print(f"❌ Error uploading to Supabase: {e}")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files")

def main():
    parser = argparse.ArgumentParser(description="Extract Data from Whiteboards via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    args = parser.parse_args()

    course_id = args.course_id

    if MODAL_WHITEBOARD_API_URL == "https://<YOUR-MODAL-WORKSPACE>--whiteboard-ai-backend-fastapi-app.modal.run/process-whiteboards":
        print("⚠️ PLEASE DEPLOY MODAL API FIRST AND UPDATE 'MODAL_WHITEBOARD_API_URL'!")
        return

    processor = CloudWhiteboardProcessor(course_id=course_id)

    try:
        if not processor.download_whiteboards():
            return
        if not processor.process_via_modal():
            return
        processor.upload_results()
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
