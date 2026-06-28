import modal
import os
import json
import re
import time
import requests
from typing import List, Dict

# --- 1. Define the Modal Environment ---
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "sentence-transformers", "nltk", "requests", "openai",
    "markdown", "fastapi[standard]", "python-multipart"
)

app = modal.App("audio-notes-generator")

# ==============================================================================
# --- HELPER FUNCTIONS: AUDIO ALIGNMENT ---
# ==============================================================================

class DynamicLectureAligner:
    def __init__(self, semantic_model, threshold=0.30, dynamic_weight=0.20):
        self.model = semantic_model
        self.threshold = threshold
        self.dynamic_weight = dynamic_weight

    def align(self, agenda_items: List[str], sentences: List[str]) -> Dict[str, List[str]]:
        import torch
        import torch.nn.functional as F
        from sentence_transformers import util
        
        if not agenda_items or not sentences:
            return {}

        alignment_results = {item: [] for item in agenda_items}

        # Encode everything
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        # .clone() prevents runtime errors when modifying tensors in-place
        current_agenda_embeddings = self.model.encode(agenda_items, convert_to_tensor=True).clone()

        for i, sentence in enumerate(sentences):
            current_sent_embedding = sentence_embeddings[i]
            scores = util.cos_sim(current_sent_embedding, current_agenda_embeddings)[0]
            max_score, max_idx = torch.max(scores, dim=0)
            
            if max_score.item() >= self.threshold:
                best_idx = max_idx.item()
                matched_topic = agenda_items[best_idx]
                alignment_results[matched_topic].append(sentence)

                # Dynamic Update Logic
                old_vec = current_agenda_embeddings[best_idx]
                new_vec = (old_vec * (1 - self.dynamic_weight)) + (current_sent_embedding * self.dynamic_weight)
                new_vec = F.normalize(new_vec, p=2, dim=0)
                current_agenda_embeddings[best_idx] = new_vec

        return alignment_results

# ==============================================================================
# --- HELPER FUNCTIONS: CLEANING & HTML ---
# ==============================================================================

def clean_batch_output(text: str) -> str:
    # Remove Markdown Code Fences (```markdown or ```)
    text = re.sub(r'^```markdown\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)

    lines = text.split('\n')
    cleaned_lines = []
    
    ignore_patterns = [
        r"^Here (is|are) the", r"^Sure,", r"^Certainly", r"^Below is the",
        r"^\(End of", r"^These notes", r"^---", r"^\*?Note:"
    ]
    
    for line in lines:
        if any(re.search(pat, line, re.IGNORECASE) for pat in ignore_patterns): continue
        if "omitted for clarity" in line or "Irrelevant digressions" in line: continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()

def latex_table_to_html(latex_text):
    inner_content = re.search(r'\\begin\{tabular\}\{.*?\}(.*?)\\end\{tabular\}', latex_text, re.DOTALL)
    if not inner_content: return latex_text
    rows = inner_content.group(1).strip().split(r'\\')
    html_rows = []
    for row in rows:
        if not row.strip(): continue
        cells = row.split('&')
        html_cells = []
        for cell in cells:
            cell = cell.replace(r'\hline', '')
            cell = re.sub(r'\\multicolumn\{.*?\}\{.*?\}\{(.*?)\}', r'\1', cell)
            cell = re.sub(r'\\textbf\{(.*?)\}', r'<b>\1</b>', cell)
            cell = cell.replace('{', '').replace('}', '').strip()
            if cell: html_cells.append(f"<td>{cell}</td>")
        if html_cells: html_rows.append(f"<tr>{''.join(html_cells)}</tr>")
    return f"<table>{''.join(html_rows)}</table>"

def fix_list_spacing(md_text: str) -> str:
    lines = md_text.split('\n')
    fixed_lines = []
    list_pattern = re.compile(r'^\s*([\*\-\+]|\d+\.)\s+')
    for i, line in enumerate(lines):
        is_list_item = bool(list_pattern.match(line))
        if i > 0:
            prev_line = fixed_lines[-1]
            if is_list_item and prev_line.strip() != '' and not bool(list_pattern.match(prev_line)):
                fixed_lines.append('')
        fixed_lines.append(line)
    return '\n'.join(fixed_lines)

# ==============================================================================
# --- MODAL GPU EXECUTION ---
# ==============================================================================

@app.cls(gpu="T4", image=image, scaledown_window=300, timeout=3600)
class AudioNotesGeneratorGPU:
    @modal.enter()
    def setup(self):
        import nltk
        import torch
        from sentence_transformers import SentenceTransformer
        
        for corpus in ['punkt', 'punkt_tab']:
            try: nltk.data.find(f'tokenizers/{corpus}')
            except LookupError: nltk.download(corpus, quiet=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("⏳ Loading semantic model (all-mpnet-base-v2)...")
        self.embedder = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        print("✅ Model loaded.")

    @modal.method()
    def process(self, agenda_text: str, transcript_text: str, supabase_url: str, supabase_key: str, deepseek_api_key: str, course_id: str):
        from nltk.tokenize import sent_tokenize
        from openai import OpenAI
        import markdown
        import torch
        
        llm_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

        print(f"\n🚀 PHASE 1: AUDIO ALIGNMENT")
        
        # Parse Agenda
        agenda_lines = agenda_text.split('\n')
        agenda_items = []
        for line in agenda_lines:
            line = line.strip()
            if not line: continue
            clean_text = re.sub(r'^[\d\.\-\)\s]+', '', line)
            if clean_text: agenda_items.append(clean_text)

        # Tokenize Transcript
        sentences = [s.strip() for s in sent_tokenize(transcript_text) if s.strip()]

        # Align
        aligner = DynamicLectureAligner(self.embedder, threshold=0.30, dynamic_weight=0.20)
        alignment_results = aligner.align(agenda_items, sentences)

        print(f"\n🚀 PHASE 2: RAG & DEEPSEEK EXPLANATIONS")

        def search_supabase(query: str, top_k: int = 3) -> List[Dict]:
            with torch.no_grad():
                query_embedding = self.embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
                if self.device == "cuda": query_embedding = query_embedding.cpu()
                query_vector = query_embedding.numpy()[0].tolist()

            try:
                rpc_params = {
                    "query_embedding": query_vector,
                    "match_course_id": course_id,
                    "match_count": top_k,
                    "similarity_threshold": 0.3
                }
                response = requests.post(
                    f"{supabase_url}/rest/v1/rpc/search_similar_chunks",
                    headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}", "Content-Type": "application/json", "Prefer": "return=representation"},
                    json=rpc_params
                )
                if response.status_code == 200: return response.json()
            except Exception as e: print(f"⚠️ RPC search failed: {e}")
            return []

        def format_rag_context(results: List[Dict]) -> str:
            if not results: return "No relevant textbook content found."
            return "\n".join(f"- {r.get('content', '').strip()}" for r in results)

        EXPLANATION_PROMPT = """You are an AI teaching assistant.
Your task is to concisely explain a specific topic discussed in a lecture using the provided background materials.

**Topic:**
{topic}

**Lecture Transcript (Context):**
{transcript}

**Background Knowledge:**
{rag_context}

**Instructions:**
1. Provide a direct, engaging explanation of the concept (2-3 sentences).
2. Synthesize the transcript with the detailed knowledge from the background context.
3. Keep the tone engaging and educational. Do not repeat the text verbatim.
4. Speak directly to the student.

**Explanation:**"""

        topic_blocks = []
        for topic, sens in alignment_results.items():
            text = " ".join(sens)
            if not text or len(text.strip()) < 10: continue

            # Search Supabase for the topic + transcript context
            rag_ctx = format_rag_context(search_supabase(f"{topic} {text}", top_k=3))
            user_prompt = EXPLANATION_PROMPT.format(topic=topic, transcript=text, rag_context=rag_ctx)

            explanation = ""
            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=300, temperature=0.4
                )
                explanation = response.choices[0].message.content.strip()
            except Exception as e:
                explanation = f"LLM Error: {str(e)}"
            time.sleep(0.4)

            # Build the structured block including the new explanation
            block = (
                f"<TOPIC title='{topic}'>\n"
                f"  <RAW_TRANSCRIPT>{text}</RAW_TRANSCRIPT>\n"
                f"  <INSTRUCTOR_ELABORATION>{explanation}</INSTRUCTOR_ELABORATION>\n"
                f"</TOPIC>"
            )
            topic_blocks.append(block)

        print(f"\n🚀 PHASE 3: DEEPSEEK SUMMARIZATION ({len(topic_blocks)} Topics)")

        BATCH_PROMPT = """
You are an expert Professor compiling lecture notes from a raw audio transcript.
**TASK:** Convert the raw transcript segments and generated explanations below into clean, academic Markdown notes.

**STRICT GUIDELINES FOR AUDIO CLEANING:**
1. **REMOVE IRRELEVANCE:** Aggressively filter out:
   - Classroom logistics (attendance, grades, deadlines).
   - Personal anecdotes unrelated to the subject (e.g., "I cried when I graduated", "people of my age are cynical").
   - Digressions about the projector, mic, or other technical issues.
   - Jokes or off-topic banter.
2. **Focus on Concepts:** Only retain the academic definitions, theories, and relevant examples. Seamlessly integrate the `<INSTRUCTOR_ELABORATION>` to enrich the notes.
3. **Structure:** - Use the provided `<TOPIC>` titles as your Markdown Headers (## Title).
   - Use bullet points for key takeaways.
   - Use bold text for terms and definitions.
4. **Flow:** Write in a professional, objective tone. Do not use "The speaker said" or "He mentioned". State the facts directly.

**INPUT DATA:**
{context_stream}

**OUTPUT FORMAT:**
- Start directly with the Markdown notes. 
- Do NOT include introductory text like "Here are the notes".
- Do NOT include concluding remarks or meta-notes like "*Note: Irrelevant info removed*".
- Do NOT wrap the output in markdown code blocks (no ```markdown).
"""

        full_notes = "# Lecture Notes\n\n"
        BATCH_SIZE = 3
        total_batches = (len(topic_blocks) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(topic_blocks), BATCH_SIZE):
            batch = topic_blocks[i : i + BATCH_SIZE]
            print(f"⏳ Processing Batch {(i // BATCH_SIZE) + 1}/{total_batches}...")
            
            merged_batch = "\n\n".join(batch)
            prompt = BATCH_PROMPT.format(context_stream=merged_batch)

            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000, temperature=0.2, stream=False
                )
                raw_output = response.choices[0].message.content.strip()
                batch_notes = clean_batch_output(raw_output)
                full_notes += f"\n\n{batch_notes}"
            except Exception as e:
                print(f"⚠️ Error in batch: {e}")
            time.sleep(1)

        full_notes = re.sub(r'\n{3,}', '\n\n', full_notes).strip()

        # ==========================================================
        # PHASE 4: HTML CONVERSION
        # ==========================================================
        print("\n🚀 PHASE 4: HTML CONVERSION")
        text = fix_list_spacing(full_notes)
        text = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})', lambda m: latex_table_to_html(m.group(0)), text, flags=re.DOTALL)

        math_blocks = {}
        def replace_math(match):
            key = f"MATHBLOCKPLACEHOLDER{len(math_blocks)}END"
            math_blocks[key] = match.group(0)
            return key

        text = re.sub(r'(\$\$.*?\$\$)', replace_math, text, flags=re.DOTALL)
        text = re.sub(r'(\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\})', replace_math, text, flags=re.DOTALL)
        text = re.sub(r'(?<!\\)\$(?!\s)(?:\\.|[^$\\\n])+\$', replace_math, text)

        import markdown
        html_body = markdown.markdown(text, extensions=['extra', 'fenced_code', 'tables', 'sane_lists'])

        for key, value in math_blocks.items():
            html_body = html_body.replace(key, value)

        css = """
        <style>
            :root { --fg-color: #37352f; --bg-color: #ffffff; --gray-bg: #f7f6f3; --border-color: #e0e0e0; --link-color: #0b6e99; --code-bg: #f7f6f3; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg-color); background-color: var(--bg-color); line-height: 1.6; font-size: 17px; max-width: 900px; margin: 0 auto; padding: 40px 20px; }
            h1, h2, h3, h4, h5, h6 { color: #202020; font-weight: 600; line-height: 1.3; }
            h1 { font-size: 2.2em; margin-top: 2em; margin-bottom: 0.5em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.2em; letter-spacing: -0.02em;}
            h2 { font-size: 1.8em; margin-top: 1.8em; margin-bottom: 0.5em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.2em;}
            h3 { font-size: 1.4em; margin-top: 1.5em; margin-bottom: 0.5em; }
            p { margin-bottom: 1.2em; }
            ul, ol { margin-top: 0.5em; margin-bottom: 1.2em; padding-left: 2em; }
            li { margin-bottom: 0.4em; }
            li > p { margin-top: 0.2em; margin-bottom: 0.2em; }
            li > ul, li > ol { margin-top: 0.4em; }
            strong { font-weight: 600; color: #111; }
            em { font-style: italic; }
            a { color: var(--link-color); text-decoration: none; }
            hr { border: 0; border-top: 1px solid var(--border-color); margin: 2.5em 0; }
            blockquote { background-color: var(--gray-bg); border-left: 4px solid #4a4a4a; padding: 16px 20px; margin: 1.5em 0; border-radius: 0 4px 4px 0; color: #555; }
            figure { margin: 2em 0; text-align: center; }
            img { display: block; margin: 20px auto; max-width: 70%; max-height: 400px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
            table { border-collapse: collapse; width: 100%; margin: 2em 0; font-size: 16px; border: 1px solid var(--border-color); }
            th, td { border: 1px solid var(--border-color); padding: 12px 16px; vertical-align: top; text-align: left; }
            th { background-color: var(--gray-bg); font-weight: 600; }
            tr:nth-child(even) { background-color: #fafafa; }
            mjx-container { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
            pre { background-color: var(--code-bg); padding: 20px; border-radius: 6px; overflow-x: auto; margin: 1.5em 0; border: 1px solid var(--border-color); }
            code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; background-color: rgba(135,131,120,0.15); color: #EB5757; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }
            pre > code { background-color: transparent; padding: 0; color: #333; font-size: 14px; border: none; }
        </style>
        """

        mathjax_script = """
        <script>
        MathJax = {
          tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true, packages: {'[+]': ['noerrors', 'noundefined']} },
          options: { ignoreHtmlClass: 'tex2jax_ignore', processHtmlClass: 'tex2jax_process' },
          svg: { fontCache: 'global' }
        };
        </script>
        <script type="text/javascript" id="MathJax-script" async src="[https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js](https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js)"></script>
        """

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Lecture Notes</title>
    {css}
    {mathjax_script}
</head>
<body>
    {html_body}
</body>
</html>"""

        print("✅ Pipeline Complete! Returning Markdown and HTML to Azure.")
        return {
            "markdown": full_notes,
            "html": full_html.strip()
        }

# --- Web API Endpoint ---
@app.function(image=image, timeout=3600)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Form
    from fastapi.responses import JSONResponse
    web_app = FastAPI()

    @web_app.post("/generate-audio-notes")
    async def generate_audio_notes(
        agenda_text: str = Form(...),
        audio_transcript: str = Form(...),
        supabase_url: str = Form(...),
        supabase_key: str = Form(...),
        deepseek_api_key: str = Form(...),
        course_id: str = Form(...)
    ):
        processor = AudioNotesGeneratorGPU()
        result = processor.process.remote(
            agenda_text, audio_transcript, supabase_url, supabase_key, deepseek_api_key, course_id
        )
        return JSONResponse(content=result)

    return web_app