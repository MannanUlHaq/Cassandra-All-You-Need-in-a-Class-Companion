import modal
import os
import json
import re
import tempfile
import shutil
from typing import List, Dict
from pathlib import Path

# --- 1. Define the Modal Environment ---
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "libgl1", "libglib2.0-0", "build-essential" # Required for OpenCV
).pip_install(
    "transformers", "torch", "torchvision", "accelerate", "opencv-python-headless", 
    "pillow", "huggingface_hub", "fastapi[standard]", "python-multipart"
)

app = modal.App("whiteboard-ai-backend")

# Connect to the exact same volume we created for the slides!
model_volume = modal.Volume.from_name("finetuned-models-vol", create_if_missing=True)

# The local path where the model will live on the volume
QWEN_LOCAL_PATH = "/models/Qwen3-VL-4B-Instruct"

# ==============================================================================
# -------------------- PRE-DEPLOYMENT: CACHE THE MODEL -------------------------
# ==============================================================================
@app.function(image=image, volumes={"/models": model_volume}, timeout=3600)
def download_qwen_model(hf_token: str):
    """
    Run this ONCE from your local terminal to download the model into the volume:
    modal run modal_whiteboard_api.py::download_qwen_model --hf-token YOUR_HF_TOKEN
    """
    from huggingface_hub import snapshot_download
    print(f"📥 Downloading Qwen3-VL-4B-Instruct to {QWEN_LOCAL_PATH}...")
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-4B-Instruct", 
        local_dir=QWEN_LOCAL_PATH, 
        token=hf_token
    )
    print("✅ Model successfully cached to Modal Volume!")

# ==============================================================================
# -------------------- PIPELINE LOGIC (Your exact code) ------------------------
# ==============================================================================
def preprocess(img_input):
    """Smart preprocessing that adapts to raw vs. already processed images."""
    import cv2
    import numpy as np

    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()

    if img is None:
        return None

    # Convert to grayscale to evaluate the standard deviation (contrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)

    # SMART CHECK: If standard deviation is high (>75), the image is likely 
    # already preprocessed/binarized (like Image 2). Skip heavy processing!
    if std_dev > 75:
        return img 

    # LIGHT ENHANCEMENT: For raw whiteboards (like Image 1)
    # Color balance and slight contrast enhancement without aggressive CLAHE
    enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    
    # Simple contrast stretch
    alpha = 1.2 # Contrast control
    beta = 10   # Brightness control
    adjusted = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return adjusted

def clean_json_string(s):
    import re
    
    # 1. Safely extract JSON between markdown code blocks without breaking IDEs
    marker = "`" * 3
    pattern = marker + r"(?:json)?\s*(.*?)\s*" + marker
    json_match = re.search(pattern, s, re.DOTALL)
    if json_match:
        s = json_match.group(1)
        
    # 2. Find the true JSON boundaries to ignore stray text
    start_idx = s.find('{')
    end_idx = s.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        s = s[start_idx:end_idx+1]
        
    # 3. Handle LaTeX backslashes carefully
    # Escape any backslash that is NOT already followed by a valid JSON escape char
    # Valid JSON escapes: " \ / b f n r t u
    s = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', s)
    
    return s.strip()

# ==============================================================================
# ------------------------- MODAL GPU EXECUTION --------------------------------
# ==============================================================================
@app.cls(gpu="A10G", image=image, volumes={"/models": model_volume}, scaledown_window=300, timeout=3600)
class WhiteboardProcessorGPU:
    @modal.method()
    def process(self, image_data_list: List[Dict]):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        import cv2

        print(f"🤖 Loading model from Volume: {QWEN_LOCAL_PATH}...")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                QWEN_LOCAL_PATH, 
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="eager"
            ).eval()
            processor = AutoProcessor.from_pretrained(QWEN_LOCAL_PATH)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return {"error": str(e)}

        all_results = []
        temp_dir = tempfile.mkdtemp()

        try:
            for idx, img_data in enumerate(image_data_list):
                filename = img_data['filename']
                img_bytes = img_data['bytes']
                
                print(f"\n--- Processing Image {idx+1}/{len(image_data_list)}: {filename} ---")
                
                # Write bytes to temp file for OpenCV
                temp_img_path = os.path.join(temp_dir, filename)
                with open(temp_img_path, "wb") as f:
                    f.write(img_bytes)

                try:
                    raw_img = cv2.imread(temp_img_path)
                    if raw_img is None:
                        print(f"Error: Image failed to load -> {filename}")
                        continue

                    # Preprocessing
                    preprocessed_img = preprocess(raw_img)
                    image_pil = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))

                    # Format prompt with correct index
                    PROMPT_TEXT = f"""You are an expert OCR engine processing whiteboard images. Extract the visual content exactly as it appears.

CRITICAL INSTRUCTION: You must output ONLY a valid, strictly formatted JSON object. Do not output any conversational text, introductory text, or explanatory sections outside of the JSON block.

If you encounter math or equations, extract them as LaTeX. You MUST double-escape all backslashes in your JSON strings (e.g., output "\\\\frac{{1}}{{2}}" instead of "\\frac{{1}}{{2}}").

Your output MUST exactly follow this schema:
{{
  "image_index": {idx+1},
  "structured_elements": [
    {{
      "original_id": "<unique_id_for_element>",
      "content_type": "<text|table|equation|figure>",
      "content": "<exact extracted content>",
      "reading_order": <integer_starting_from_1>,
      "metadata": {{
        "image_path": "{filename}",
        "bbox": [x_min, y_min, x_max, y_max],
        "raw_type": "<original detected type>"
      }}
    }}
  ]
}}
"""
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_pil},
                                {"type": "text", "text": PROMPT_TEXT}
                            ]
                        }
                    ]

                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                    ).to(model.device)

                    input_len = inputs["input_ids"].shape[-1]

                    with torch.inference_mode():
                        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                        generated_ids = output_ids[0][input_len:]

                    output_text = processor.decode(generated_ids, skip_special_tokens=True)
                    cleaned_json = clean_json_string(output_text)
                    
                    try:
                        # strict=False allows parsing multi-line text blocks with literal newlines
                        json_obj = json.loads(cleaned_json, strict=False)
                        all_results.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON Parse Error for {filename}: {e}\nRaw output: {cleaned_json}")
                        all_results.append({"image_index": idx + 1, "error": "Parse Error", "raw_output": output_text})

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

            return {"status": "success", "results": all_results}

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            # Free up memory explicitly
            del model, processor
            torch.cuda.empty_cache()


# --- 3. Web API Endpoint ---
@app.function(image=image, timeout=3600)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File
    web_app = FastAPI()

    @web_app.post("/process-whiteboards")
    async def process_whiteboards(images: List[UploadFile] = File(...)):
        # Read files into memory to cross the Modal remote boundary
        image_data_list = []
        for img in images:
            image_data_list.append({
                "filename": img.filename,
                "bytes": await img.read()
            })
        
        processor = WhiteboardProcessorGPU()
        result = processor.process.remote(image_data_list)
        return result

    return web_app