import modal
import os
import uuid
import subprocess
from pathlib import Path

# 1. Environment dependencies (ffmpeg is key here)
image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "ffmpeg", "build-essential", "libsndfile1"
).pip_install(
    "faster-whisper==1.0.3",
    "resemblyzer==0.1.4",
    "torch==2.2.1",
    "numpy==1.26.4",
    "soundfile==0.12.1",
    "fastapi[standard]",
    "python-multipart"
)

app = modal.App("audio-ai-backend")

# 2. Define the GPU Class
@app.cls(gpu="T4", image=image, container_idle_timeout=300)
class AudioProcessorGPU:
    @modal.enter()
    def load_models(self):
        import torch
        from faster_whisper import WhisperModel
        from resemblyzer import VoiceEncoder
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading models into {self.device} VRAM...")
        
        # Using float16 on GPU provides a massive speedup for large-v3
        self.whisper_model = WhisperModel("large-v3", device=self.device, compute_type="float16")
        self.encoder = VoiceEncoder(device=self.device)

    def _normalize_audio(self, input_bytes: bytes) -> str:
        """Converts any raw audio bytes into a strict, clean, 16kHz, 16-bit mono WAV file using ffmpeg."""
        import tempfile
        
        # Write incoming raw bytes to a temp file
        raw_fd, raw_path = tempfile.mkstemp()
        with os.fdopen(raw_fd, 'wb') as f:
            f.write(input_bytes)
        
        # Create a destination path for the clean WAV
        clean_fd, clean_path = tempfile.mkstemp(suffix=".wav")
        os.close(clean_fd)
        
        try:
            # Force strictly standard PCM format for SoundFile/Resemblyzer compatibility
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path, 
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
                clean_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path) # Always clean up the raw file
                
        return clean_path

    @modal.method()
    def process(self, teacher_bytes: bytes, lecture_bytes: bytes, similarity_threshold: float = 0.50):
        from resemblyzer import preprocess_wav
        import soundfile as sf
        import numpy as np
        
        t_path = None
        l_path = None

        try:
            print("🔧 Normalizing audio formats via ffmpeg...")
            t_path = self._normalize_audio(teacher_bytes)
            l_path = self._normalize_audio(lecture_bytes)

            print("🗣️ Generating teacher embedding...")
            teacher_wav = preprocess_wav(Path(t_path))
            teacher_embed = self.encoder.embed_utterance(teacher_wav)

            print("📝 Transcribing lecture audio...")
            segments, info = self.whisper_model.transcribe(
                l_path,
                beam_size=5,
                task="translate",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            print("🔍 Filtering transcription by speaker...")
            teacher_text_segments = []
            
            # This will now successfully read the file because it is a strict, clean WAV!
            audio, sr = sf.read(l_path)

            for segment in segments:
                start_sample = int(segment.start * sr)
                end_sample = int(segment.end * sr)
                audio_segment = audio[start_sample:end_sample]

                temp_seg_path = f"/tmp/seg_{uuid.uuid4().hex}.wav"
                try:
                    sf.write(temp_seg_path, audio_segment, sr)
                    segment_wav = preprocess_wav(Path(temp_seg_path))
                    
                    if len(segment_wav) >= 8000:
                        segment_embed = self.encoder.embed_utterance(segment_wav)
                        similarity = np.dot(teacher_embed, segment_embed)
                        
                        if similarity >= similarity_threshold:
                            teacher_text_segments.append(segment.text.strip())
                finally:
                    if os.path.exists(temp_seg_path):
                        os.remove(temp_seg_path)

            return {
                "teacher_text": " ".join(teacher_text_segments),
                "language": info.language,
                "duration": info.duration
            }
        
        finally:
            # Clean up the normalized temp files from the GPU container
            if t_path and os.path.exists(t_path):
                os.remove(t_path)
            if l_path and os.path.exists(l_path):
                os.remove(l_path)

# 3. Define the Web API endpoint
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File
    
    web_app = FastAPI()

    @web_app.post("/process-audio")
    async def process_audio(
        teacher_audio: UploadFile = File(...),
        lecture_audio: UploadFile = File(...)
    ):
        # Read the files into memory as bytes inside the Web Container
        t_bytes = await teacher_audio.read()
        l_bytes = await lecture_audio.read()

        # Send the raw bytes over the network to the GPU Container
        processor = AudioProcessorGPU()
        result = processor.process.remote(t_bytes, l_bytes)
        
        return result

    return web_app