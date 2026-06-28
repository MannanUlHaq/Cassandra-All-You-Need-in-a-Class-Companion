import argparse
import os
import json
import tempfile
import requests
import zipfile
import shutil
from supabase import create_client, Client

# === Configuration ===
SUPABASE_URL = "https://lchcfsgexlkjxrhfjrna.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Your DeepSeek API Key needed by Modal
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# The new Modal API URL for Slides (Update this if your Modal deployment URL changes)
MODAL_SLIDES_API_URL = "https://cassandra-classcompanion--slides-ai-backend-fastapi-app.modal.run/process-slides"

class CloudSlidesProcessor:
    def __init__(self, course_id):
        self.course_id = course_id
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_path = None
        self.pptx_path = None
        self.extracted_dir = None
        self.output_json_path = None

    def download_slides(self):
        """Downloads the PDF and PPTX slides from Supabase lecture_slides bucket"""
        print(f"☁️ Downloading slides for course: {self.course_id}")

        pdf_filename = f"{self.course_id}_slides.pdf"
        pptx_filename = f"{self.course_id}_slides.pptx"

        try:
            # 1. Download PDF
            print(f"📥 Downloading {pdf_filename}...")
            pdf_data = self.supabase.storage.from_('lecture_slides').download(pdf_filename)
            self.pdf_path = os.path.join(self.temp_dir, pdf_filename)
            with open(self.pdf_path, "wb") as f:
                f.write(pdf_data)

            # 2. Download PPTX
            print(f"📥 Downloading {pptx_filename}...")
            pptx_data = self.supabase.storage.from_('lecture_slides').download(pptx_filename)
            self.pptx_path = os.path.join(self.temp_dir, pptx_filename)
            with open(self.pptx_path, "wb") as f:
                f.write(pptx_data)

            print("✅ Successfully downloaded both PDF and PPTX files.")
            return True

        except Exception as e:
            print(f"❌ Error downloading slides from Supabase: {e}")
            print("Make sure BOTH the .pdf and .pptx files exist in the 'lecture_slides' bucket!")
            return False

    def process_slides_via_modal(self):
        """Send the downloaded slides to Modal GPU API for heavy processing"""
        print("🚀 Sending slides to Modal GPU API for decomposition (This may take a few minutes)...")
        response = None
        try:
            with open(self.pdf_path, 'rb') as pdf_file, open(self.pptx_path, 'rb') as pptx_file:
                response = requests.post(
                    MODAL_SLIDES_API_URL,
                    files={
                        'pdf_file': pdf_file,
                        'pptx_file': pptx_file
                    },
                    data={
                        'deepseek_api_key': DEEPSEEK_API_KEY
                    },
                    timeout=1800  # 30-minute timeout for large slide decks and heavy ML models
                )

            response.raise_for_status()

            # Save the ZIP returned by Modal
            zip_path = os.path.join(self.temp_dir, f"{self.course_id}_output.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)

            # Extract the ZIP
            self.extracted_dir = os.path.join(self.temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extracted_dir)

            # Ensure JSON exists
            self.output_json_path = os.path.join(self.extracted_dir, "result.json")
            if not os.path.exists(self.output_json_path):
                print("❌ Error: result.json not found inside the ZIP.")
                return False

            print("✅ Slides completely processed and ZIP extracted!")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ API error from Modal: {e}")
            if response is not None:
                print(f"Modal Error Details: {response.text}")
            return False

    def clear_old_slides_data(self):
        """Removes existing processed slides JSON and media for this course to prevent stale data."""
        print(f"\n🧹 Clearing old slides data for course: {self.course_id}...")
        try:
            # List items in the root of the course folder
            items = self.supabase.storage.from_('decomposed_slides').list(self.course_id)
            if items:
                files_to_remove = []
                for item in items:
                    if item['name'] == '.emptyFolderPlaceholder':
                        continue
                        
                    sub_path = f"{self.course_id}/{item['name']}"
                    
                    # Try to list inside (in case it's a folder like 'media/')
                    sub_items = self.supabase.storage.from_('decomposed_slides').list(sub_path)
                    
                    if sub_items and len(sub_items) > 0:
                        for sub_item in sub_items:
                            if sub_item['name'] != '.emptyFolderPlaceholder':
                                files_to_remove.append(f"{sub_path}/{sub_item['name']}")
                    else:
                        files_to_remove.append(sub_path)
                
                if files_to_remove:
                    self.supabase.storage.from_('decomposed_slides').remove(files_to_remove)
                    print(f"✅ Cleared {len(files_to_remove)} old slide asset(s).")
            else:
                print("✅ No old slides data found to clear.")
        except Exception as e:
            print(f"⚠️ Could not clear old slides data (might not exist yet): {e}")

    def upload_results(self):
        """Upload the extracted media files and final structured JSON back to Supabase"""
        if not self.extracted_dir or not os.path.exists(self.extracted_dir):
            print("❌ No extracted files found to upload.")
            return

        # 1. Clear out old files first
        self.clear_old_slides_data()

        print(f"\n📤 Uploading media and JSON to Supabase 'decomposed_slides/{self.course_id}/'...")

        # 2. Upload Media Files (images, tables, equations)
        for root, dirs, files in os.walk(self.extracted_dir):
            for file in files:
                if file == "result.json":
                    continue  # We will upload the JSON separately at the end

                local_path = os.path.join(root, file)

                # Get relative path (e.g., 'media/slide1_e1.png')
                rel_path = os.path.relpath(local_path, self.extracted_dir)
                # Ensure forward slashes for URLs
                rel_path = rel_path.replace("\\", "/")

                # Prepend the course_id to create an isolated folder in Supabase
                destination_path = f"{self.course_id}/{rel_path}"

                # Read MIME type automatically or set default for images
                content_type = "image/png" if file.endswith(".png") else "application/octet-stream"

                try:
                    with open(local_path, "rb") as f:
                        self.supabase.storage.from_('decomposed_slides').upload(
                            path=destination_path,
                            file=f,
                            file_options={"content-type": content_type, "upsert": "true"}
                        )
                    print(f"  ⬆️ Uploaded {rel_path}")
                except Exception as e:
                    print(f"  ❌ Error uploading {rel_path}: {e}")

        # 3. Upload the JSON File
        try:
            with open(self.output_json_path, "rb") as f:
                destination_path = f"{self.course_id}/{self.course_id}_Slides_structured.json"
                self.supabase.storage.from_('decomposed_slides').upload(
                    path=destination_path,
                    file=f,
                    file_options={"content-type": "application/json", "upsert": "true"}
                )
            print(f"✅ JSON successfully uploaded: decomposed_slides/{destination_path}")
        except Exception as e:
            print(f"❌ Error uploading JSON to Supabase: {e}")

    def cleanup(self):
        """Remove temporary directory and files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Cleaned up temporary files")


def main():
    parser = argparse.ArgumentParser(description="Process Slides via Modal")
    parser.add_argument("--course_id", required=True, help="The ID of the course")
    args = parser.parse_args()

    course_id = args.course_id

    processor = CloudSlidesProcessor(course_id=course_id)

    try:
        if not processor.download_slides():
            return
        if not processor.process_slides_via_modal():
            return
        processor.upload_results()
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
