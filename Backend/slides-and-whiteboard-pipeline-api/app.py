from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
import os

app = FastAPI(title="Cassandra Hybrid (Slides+Whiteboard) Pipeline")

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
    mode: str = "slides_and_whiteboard"

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

async def process_hybrid_pipeline(course_id: str, mode: str):
    """Background task: Runs Audio, Slides, and Whiteboard sequentially to avoid GPU OOM, then Notes."""
    print(f"🚀 [PIPELINE STARTED] Course: {course_id} | Mode: {mode}")

    # 1. Process Audio
    print("⏳ [1/4] Running Audio Extractor...")
    audio_status = await run_python_script("audio_processing.py", ["--course_id", course_id, "--mode", mode])
    
    # 2. Process Slides
    print("⏳ [2/4] Running Slides Extractor...")
    slides_status = await run_python_script("slides_processing.py", ["--course_id", course_id])

    # 3. Process Whiteboards
    print("⏳ [3/4] Running Whiteboard Extractor...")
    wb_status = await run_python_script("whiteboard_extractor.py", ["--course_id", course_id])

    # Check if all extractors exited successfully
    if audio_status == 0 and slides_status == 0 and wb_status == 0:
        print("✅ [4/4] Extractors finished successfully. Starting Combined Notes Generation...")
        
        # Trigger the combined notes generator script
        # Note: Ensure the script name matches your actual python file!
        notes_status = await run_python_script("slides+whiteboard_notes.py", ["--course_id", course_id])
        
        if notes_status == 0:
            print(f"🎉 [PIPELINE COMPLETE] Hybrid Notes successfully generated for {course_id}")
        else:
            print(f"❌ Notes generation failed for {course_id}")
    else:
        print("❌ Sequential extraction failed. Halting pipeline. Notes will not be generated.")

@app.post("/api/end_session")
async def end_session(payload: EndSessionPayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_hybrid_pipeline, payload.course_id, payload.mode)
    
    return {
        "status": "success", 
        "message": "Hybrid lecture ended. Background processing initiated."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)