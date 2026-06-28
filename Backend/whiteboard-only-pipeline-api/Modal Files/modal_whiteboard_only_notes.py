import modal
import os
import json
import re
import time
import requests
from typing import List, Dict

# --- 1. Define the Modal Environment ---
# Removed beautifulsoup4 - we are using the robust Markdown/Regex logic from the Slides pipeline!
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "sentence-transformers", "nltk", "requests", "openai",
    "markdown", "fastapi[standard]", "python-multipart"
)

app = modal.App("whiteboard-notes-generator")

# ==============================================================================
# --- HELPER FUNCTIONS & CLASSES ---
# ==============================================================================

class WhiteboardAudioIntegrator:
    def __init__(self, semantic_model, threshold=0.60):
        self.model = semantic_model
        self.threshold = threshold
        self.section_ids = []
        self.section_embeddings = {}

    def extract_text_elements(self, structured_data, id_key='original_id', type_key='content_type', valid_types=('text','paragraph')):
        import torch
        import torch.nn.functional as F
        self.section_ids = []
        self.section_embeddings = {}
        
        elements = structured_data.get('structured_elements', []) or []
        for element in elements:
            e_type = element.get(type_key)
            e_id = element.get(id_key)
            if e_type not in valid_types:
                continue
            
            content = (element.get('content') or '').strip()
            if content and e_id is not None:
                self.section_ids.append(e_id)
                emb = self.model.encode(content, convert_to_tensor=True)
                self.section_embeddings[e_id] = F.normalize(emb, p=2, dim=0)

    def match_audio_to_texts(self, sentences):
        import torch
        from sentence_transformers import util
        if not sentences or not self.section_ids:
            return {}

        assignments = {eid: [] for eid in self.section_ids}
        embeddings_list = [self.section_embeddings[eid] for eid in self.section_ids]
        section_tensor = torch.stack(embeddings_list) 

        for s in sentences:
            audio_emb = self.model.encode(s, convert_to_tensor=True)
            audio_emb = torch.nn.functional.normalize(audio_emb, p=2, dim=0)
            sims = util.cos_sim(audio_emb, section_tensor)
            max_sim, max_idx = sims[0].max(0)
            
            if max_sim.item() >= self.threshold:
                best_id = self.section_ids[max_idx.item()]
                assignments[best_id].append(s)
                
        return assignments

def format_rag_context(results: List[Dict]) -> str:
    if not results:
        return "No relevant background material found."
    return "\n".join(f"- {r.get('content','')}" for r in results)

def get_board_context(board: Dict, current_id: str) -> str:
    parts = [f"Image {board.get('image_index')}"]
    for el in board.get("structured_elements", []):
        if el.get("original_id") == current_id:
            continue
        t = el.get("content_type", "")
        c = el.get("content", "")
        parts.append(f"- {t}: {c}")
    return "\n".join(parts)

def clean_batch_output(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    ignore_patterns = [
        r"^Here (is|are) the", r"^Sure,", r"^Certainly", r"^Below is the",
        r"^\(End of", r"^These notes", r"^---", 
    ]
    for line in lines:
        if any(re.search(pat, line, re.IGNORECASE) for pat in ignore_patterns) and len(lines) > 5:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

# --- HTML Conversion Helpers (Matched Exactly to Slides Pipeline) ---
def latex_table_to_html(latex_text):
    inner_content = re.search(r'\\begin\{tabular\}\{.*?\}(.*?)\\end\{tabular\}', latex_text, re.DOTALL)
    if not inner_content: 
        return latex_text
    
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
            if cell: 
                html_cells.append(f"<td>{cell}</td>")
        if html_cells: 
            html_rows.append(f"<tr>{''.join(html_cells)}</tr>")
    
    return f"<table>{''.join(html_rows)}</table>"

def fix_list_spacing(md_text: str) -> str:
    lines = md_text.split('\n')
    fixed_lines = []
    list_pattern = re.compile(r'^\s*([\*\-\+]|\d+\.)\s+')
    
    for i, line in enumerate(lines):
        is_list_item = bool(list_pattern.match(line))
        if i > 0:
            prev_line = fixed_lines[-1]
            prev_is_empty = prev_line.strip() == ''
            prev_is_list_item = bool(list_pattern.match(prev_line))
            if is_list_item and not prev_is_empty and not prev_is_list_item:
                fixed_lines.append('')
        fixed_lines.append(line)
    return '\n'.join(fixed_lines)


# ==============================================================================
# --- MODAL GPU EXECUTION ---
# ==============================================================================

@app.cls(gpu="T4", image=image, scaledown_window=300, timeout=3600)
class WhiteboardNotesGeneratorGPU:
    @modal.enter()
    def setup(self):
        import nltk
        import torch
        from sentence_transformers import SentenceTransformer
        
        for corpus in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{corpus}')
            except LookupError:
                nltk.download(corpus, quiet=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("⏳ Loading semantic model (all-mpnet-base-v2)...")
        self.embedder = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        print("✅ Model loaded.")

    @modal.method()
    def process(self, wb_json_str: str, audio_json_str: str, supabase_url: str, supabase_key: str, deepseek_api_key: str, course_id: str):
        from nltk.tokenize import sent_tokenize
        from openai import OpenAI
        import markdown
        
        llm_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

        wb_data = json.loads(wb_json_str)
        audio_data = json.loads(audio_json_str)

        boards = []
        if isinstance(wb_data, dict) and "results" in wb_data and isinstance(wb_data["results"], list):
            boards = wb_data["results"]
        elif isinstance(wb_data, list):
            boards = wb_data
        else:
            boards = [wb_data]

        print(f"\n🚀 PHASE 1 & 2: AUDIO ALIGNMENT & RAG EXPLANATIONS (Found {len(boards)} boards)")
        
        EXPLANATION_PROMPT = """You are an AI teaching assistant. 
Your task is to explain the concept behind the text found on a whiteboard.

**Whiteboard Context:**
{board_context}

**Background Knowledge:**
{rag_context}

**Text to Explain:**
"{target_text}"

**Instructions:**
1. Provide a direct, engaging explanation of the concept (2-3 sentences).
2. Seamlessly integrate the whiteboard content with the detailed background knowledge.
3. **STRICTLY FORBIDDEN:** Do NOT use phrases like "The slide shows", "The textbook says", "The board mentions", or "According to the context".
4. Speak directly to the student about the subject matter.

**Explanation:**"""

        def search_supabase(query: str, top_k: int = 3) -> List[Dict]:
            import torch
            with torch.no_grad():
                query_embedding = self.embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)
                if self.device == "cuda": 
                    query_embedding = query_embedding.cpu()
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
            except Exception as e:
                print(f"⚠️ RPC search failed: {e}")
            return []

        for board in boards:
            idx = board.get('image_index')
            if board.get('error') or idx is None: continue

            # --- Audio Alignment ---
            audio_key = f"whiteboard_{idx}"
            raw_audio = audio_data.get(audio_key, "")
            sentences = [s.strip() for s in sent_tokenize(raw_audio) if s.strip()]

            if sentences:
                integrator = WhiteboardAudioIntegrator(self.embedder, threshold=0.55)
                integrator.extract_text_elements(board, id_key='original_id', type_key='content_type', valid_types=('text','paragraph'))
                matches = integrator.match_audio_to_texts(sentences)

                for el in board.get('structured_elements', []):
                    e_id = el.get('original_id')
                    if e_id in matches and matches[e_id]:
                        original = el.get('content', '')
                        audio_text = " ".join(matches[e_id])
                        el['content'] = f"{original}\n\n{audio_text}"

            # --- Explanations ---
            for el in board.get("structured_elements", []):
                if el.get("content_type") not in ("text", "paragraph"): continue
                text = el.get("content", "").strip()
                if len(text) < 5: continue

                board_ctx = get_board_context(board, el.get("original_id"))
                rag_ctx = format_rag_context(search_supabase(text, top_k=3))
                prompt = EXPLANATION_PROMPT.format(target_text=text, board_context=board_ctx, rag_context=rag_ctx)
                
                try:
                    res = llm_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=300
                    )
                    el["explanation"] = res.choices[0].message.content.strip()
                except Exception as e:
                    el["explanation"] = f"LLM Error: {e}"
                time.sleep(0.4)

        print("\n🚀 PHASE 3: LECTURE NOTES (SUMMARIZATION)")

        frame_list = []
        for item in boards:
            if item.get("error"): continue
            slide_num = item.get('image_index', '?')
            current_frame = f"\n<FRAME index='{slide_num}'>\n"

            elements = item.get('structured_elements', []) or []
            try: elements = sorted(elements, key=lambda e: int(e.get('reading_order', 9999)))
            except: pass

            for elem in elements:
                e_type = (elem.get('content_type') or elem.get('type') or 'text').lower()
                content = elem.get('content', '').strip()
                explanation = elem.get('explanation', '').strip()

                if e_type in ('text', 'paragraph'): current_frame += f"  <TEXT>{content}</TEXT>\n"
                elif e_type == 'image': 
                    pass # Skip visual context completely for whiteboards
                elif e_type == 'equation': 
                    # Removed the img_ref logic here so it doesn't tempt the LLM to link an image
                    current_frame += f"  <EQUATION>{content}</EQUATION>\n"
                elif e_type == 'table': current_frame += f"  <TABLE_DATA>\n{content}\n</TABLE_DATA>\n"
                else: current_frame += f"  <FRAME_TEXT>{content}</FRAME_TEXT>\n"

                if explanation: current_frame += f"  <INSTRUCTOR_ELABORATION>{explanation}</INSTRUCTOR_ELABORATION>\n"

            current_frame += "</FRAME>"
            frame_list.append(current_frame)

        BATCH_PROMPT = """
You are an expert Professor writing a textbook chapter.
**TASK:** Convert the raw whiteboard data below into clean, seamless Markdown lecture notes.

**STRICT RULES:**
1. **NO CONVERSATION:** Do not write "Here are the notes", "In this slide", "Summary", or "End of batch". Output *only* the note content. Start directly with the first Header.
2. **Seamless Flow:** Write as if this is one continuous document. If the current frames continue the topic from the PREVIOUS NOTES, continue seamlessly WITHOUT repeating the main heading.
3. **Math:** Preserve ALL derivation steps and equations using LaTeX ($...$ or $$...$$). ALWAYS leave a blank line before and after block equations ($$...$$) for proper rendering.
4. **Detail:** Do not summarize. Include every rule, example, and definition.
5. **NO IMAGES (CRITICAL):** Whiteboard notes should NOT contain any images. Do NOT generate any Markdown image tags (e.g., `![alt text](image.png)`).
6. **Formatting Mastery (CRITICAL FOR LISTS):**
   - **Un-mash Lists:** Raw text often mashes bullet points onto a single line (e.g., `* Item 1 * Item 2 * Item 3`). YOU MUST format these into proper vertical Markdown lists, with each item on a new line.
   - **Blank Lines:** ALWAYS place an empty line before starting a list, table, or code block.
   - **Emphasis:** Use **bold** (`**text**`) for important terms, quantities, and definitions.
7. **Context Awareness:** DO NOT repeat the text from the "PREVIOUS NOTES". It is only there so you know what was just discussed. Continue the document naturally.

**PREVIOUS NOTES (For context only - DO NOT REPEAT THIS):**
{previous_context}

**INPUT DATA (Frames {start}-{end}):**
{context_stream}
"""

        full_notes = ""
        last_batch_notes = ""
        BATCH_SIZE = 3
        total_batches = (len(frame_list) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(frame_list), BATCH_SIZE):
            batch = frame_list[i : i + BATCH_SIZE]
            start_idx = i + 1
            end_idx = min(i + BATCH_SIZE, len(frame_list))
            print(f"⏳ Batch { (i // BATCH_SIZE) + 1 }/{total_batches}...")
            
            merged_batch = "\n".join(batch)
            prompt = BATCH_PROMPT.format(
                start=start_idx, 
                end=end_idx, 
                previous_context=last_batch_notes if last_batch_notes else "None (Beginning of document).",
                context_stream=merged_batch
            )

            try:
                response = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.1,
                    stream=False
                )
                raw_output = response.choices[0].message.content.strip()
                batch_notes = clean_batch_output(raw_output)
                full_notes += f"\n\n{batch_notes}"
                last_batch_notes = batch_notes
            except Exception as e:
                print(f"⚠️ Error in batch {start_idx}-{end_idx}: {e}")
            time.sleep(1)

        full_notes = re.sub(r'\n{3,}', '\n\n', full_notes).strip()

        print("\n🚀 PHASE 4: HTML CONVERSION")
        
        # 1. Pre-process spacing for Markdown lists
        text = fix_list_spacing(full_notes)

        # 2. Convert LaTeX tables
        text = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})', lambda m: latex_table_to_html(m.group(0)), text, flags=re.DOTALL)

        # 3. Protect math securely (Exact Slides Logic)
        math_blocks = {}
        def replace_math(match):
            key = f"MATHBLOCKPLACEHOLDER{len(math_blocks)}END"
            math_blocks[key] = match.group(0)
            return key

        text = re.sub(r'(\$\$.*?\$\$)', replace_math, text, flags=re.DOTALL)
        text = re.sub(r'(\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\})', replace_math, text, flags=re.DOTALL)
        text = re.sub(r'(?<!\\)\$(?!\s)(?:\\.|[^$\\\n])+\$', replace_math, text)

        # 4. Markdown to HTML conversion with sane_lists
        html_body = markdown.markdown(text, extensions=['extra', 'fenced_code', 'tables', 'sane_lists'])

        # 5. Restore math blocks
        for key, value in math_blocks.items():
            html_body = html_body.replace(key, value)

        # EXACT CSS FROM SLIDES PIPELINE
        css = """
        <style>
            :root {
                --fg-color: #37352f;
                --bg-color: #ffffff;
                --gray-bg: #f7f6f3;
                --border-color: #e0e0e0;
                --link-color: #0b6e99;
                --code-bg: #f7f6f3;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                color: var(--fg-color);
                background-color: var(--bg-color);
                line-height: 1.6;
                font-size: 17px;
                max-width: 900px;
                margin: 0 auto;
                padding: 40px 20px;
            }

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
            a:hover { text-decoration: underline; }
            hr { border: 0; border-top: 1px solid var(--border-color); margin: 2.5em 0; }

            blockquote {
                background-color: var(--gray-bg);
                border-left: 4px solid #4a4a4a;
                padding: 16px 20px;
                margin: 1.5em 0;
                border-radius: 0 4px 4px 0;
                color: #555;
            }

            figure { margin: 2em 0; text-align: center; }
            img {
                display: block;
                margin: 20px auto;
                max-width: 70%;
                max-height: 400px;
                width: auto;
                height: auto;
                border-radius: 6px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }

            table { border-collapse: collapse; width: 100%; margin: 2em 0; font-size: 16px; border: 1px solid var(--border-color); }
            th, td { border: 1px solid var(--border-color); padding: 12px 16px; vertical-align: top; text-align: left; }
            th { background-color: var(--gray-bg); font-weight: 600; }
            tr:nth-child(even) { background-color: #fafafa; }

            mjx-container { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
            pre { background-color: var(--code-bg); padding: 20px; border-radius: 6px; overflow-x: auto; margin: 1.5em 0; border: 1px solid var(--border-color); }
            code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; background-color: rgba(135,131,120,0.15); color: #EB5757; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }
            pre > code { background-color: transparent; padding: 0; color: #333; font-size: 14px; border: none; }
        </style>
        """

        mathjax_script = """
        <script>
        MathJax = {
          tex: {
            inlineMath: [['$', '$']], 
            displayMath: [['$$', '$$']],
            processEscapes: true,
            packages: {'[+]': ['noerrors', 'noundefined']}
          },
          options: {
            ignoreHtmlClass: 'tex2jax_ignore',
            processHtmlClass: 'tex2jax_process'
          },
          svg: { fontCache: 'global' }
        };
        </script>
        <script type="text/javascript" id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
        </script>
        """

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whiteboard Lecture Notes</title>
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

    @web_app.post("/generate-whiteboard-notes")
    async def generate_whiteboard_notes(
        wb_json: str = Form(...),
        audio_json: str = Form(...),
        supabase_url: str = Form(...),
        supabase_key: str = Form(...),
        deepseek_api_key: str = Form(...),
        course_id: str = Form(...)
    ):
        processor = WhiteboardNotesGeneratorGPU()
        result = processor.process.remote(wb_json, audio_json, supabase_url, supabase_key, deepseek_api_key, course_id)
        return JSONResponse(content=result)

    return web_app