from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
import os

app = FastAPI(title="Cassandra Audio-Only Pipeline (Azure Version)")

# Allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class EndSessionPayload(BaseModel):
    session_id: str
    course_id: str
    mode: str = "audio_only"

async def run_python_script(script_name: str, args: list):
    """Executes a Python script as a subprocess and waits for it to finish."""
    cmd = [sys.executable, script_name] + args
    print(f"Executing: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        print(f"❌ Error running script: {script_name}\n{stderr.decode('utf-8', errors='replace')}")
    else:
        print(f"✅ Success: {script_name}\n{stdout.decode('utf-8', errors='replace')}")
        
    return process.returncode

async def process_audio_only_pipeline(course_id: str, mode: str):
    """Background task: Runs Audio Extraction, then Notes generation."""
    print(f"🚀 [PIPELINE STARTED] Course: {course_id} | Mode: {mode}")

    # 1. Run Audio Extractor
    print("⏳ Running Audio Extractor...")
    audio_status = await run_python_script("audio_processing.py", ["--course_id", course_id, "--mode", mode])

    if audio_status == 0:
        print("✅ Audio processed successfully. Starting Notes Generation...")
        
        # 2. Trigger the notes generator script
        notes_status = await run_python_script("audio_only_notes.py", ["--course_id", course_id])
        
        if notes_status == 0:
            print(f"🎉 [PIPELINE COMPLETE] Audio Notes successfully generated for {course_id}")
        else:
            print(f"❌ Notes generation failed for {course_id}")
    else:
        print("❌ Audio extraction failed. Halting pipeline. Notes will not be generated.")

@app.post("/api/end_session")
async def end_session(payload: EndSessionPayload, background_tasks: BackgroundTasks):
    # Triggers the pipeline in the background
    background_tasks.add_task(process_audio_only_pipeline, payload.course_id, payload.mode)
    
    return {
        "status": "success", 
        "message": "Audio lecture ended. Background processing initiated."
    }

if __name__ == "__main__":
    import uvicorn
    # Azure Web Apps default to port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)