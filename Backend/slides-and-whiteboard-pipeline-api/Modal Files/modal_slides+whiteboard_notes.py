import modal
import os
import json
import re
import time
import requests
import math
from typing import List, Dict, Any, Optional

# --- 1. Define the Modal Environment ---
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "sentence-transformers", "nltk", "requests", "openai",
    "markdown", "fastapi[standard]", "python-multipart"
)

app = modal.App("combined-notes-generator")

# ==============================================================================
# --- HELPER FUNCTIONS: AUDIO, RAG & CLEANING ---
# ==============================================================================

class DynamicSlideIntegrator:
    def __init__(self, semantic_model, threshold=0.50, dynamic_weight=0.20):
        self.semantic_model = semantic_model
        self.threshold = threshold
        self.dynamic_weight = dynamic_weight
        self.section_ids = []
        self.section_embeddings = {}

    def extract_text_elements(self, structured_slide_data):
        import torch
        import torch.nn.functional as F
        self.section_ids = []
        self.section_embeddings = {}
        
        elements = structured_slide_data.get('structured_elements', [])
        
        for element in elements:
            if element.get('type') != 'text':
                continue
            content = element.get('content', '').strip()
            element_id = element.get('id')
            if content and element_id:
                self.section_ids.append(element_id)
                emb = self.semantic_model.encode(content, convert_to_tensor=True)
                self.section_embeddings[element_id] = F.normalize(emb, p=2, dim=0)

    def align_dynamic(self, audio_sentences):
        import torch
        import torch.nn.functional as F
        from sentence_transformers import util
        
        if not audio_sentences or not self.section_ids:
            return {}

        assignments = {eid: [] for eid in self.section_ids}
        audio_embeddings = self.semantic_model.encode(audio_sentences, convert_to_tensor=True)

        for i, sentence in enumerate(audio_sentences):
            current_audio_emb = audio_embeddings[i]
            current_matrix = torch.stack([self.section_embeddings[eid] for eid in self.section_ids])
            scores = util.cos_sim(current_audio_emb, current_matrix)[0]
            max_score, max_idx = torch.max(scores, dim=0)
            best_score = max_score.item()
            
            if best_score >= self.threshold:
                best_match_id = self.section_ids[max_idx.item()]
                assignments[best_match_id].append(sentence)
                
                old_vec = self.section_embeddings[best_match_id]
                new_vec = (old_vec * (1 - self.dynamic_weight)) + (current_audio_emb * self.dynamic_weight)
                new_vec = F.normalize(new_vec, p=2, dim=0)
                self.section_embeddings[best_match_id] = new_vec

        return assignments

def format_rag_context(results: List[Dict]) -> str:
    if not results:
        return "No relevant textbook content found."
    return "\n".join(f"- {r.get('content', '').strip()}" for r in results)

def get_slide_context(slide: Dict, current_element_id: str) -> str:
    context_parts = [f"Slide {slide.get('slide_index', '?')}"]
    for element in slide.get('structured_elements', []):
        if element.get('id') == current_element_id: continue
        e_type = element.get('type', 'unknown')
        e_content = element.get('content', '')
        if e_type == 'text': context_parts.append(f"- Text: {e_content}")
        elif e_type == 'image': context_parts.append(f"- Image: {element.get('rationale', e_content)}")
        elif e_type == 'equation': context_parts.append(f"- Equation: {e_content}")
    return "\n".join(context_parts)

def get_board_context(board: Dict, current_id: str) -> str:
    parts = [f"Whiteboard {board.get('image_index', '?')}"]
    for el in board.get("structured_elements", []):
        if el.get("original_id") == current_id: continue
        parts.append(f"- {el.get('content_type', 'unknown')}: {el.get('content', '')}")
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
# --- HELPER FUNCTIONS: NORMALIZATION & MERGING ---
# ==============================================================================

def normalize_slides(slides_raw: Any) -> List[Dict[str,Any]]:
    if isinstance(slides_raw, dict):
        slides = slides_raw.get('slides') or slides_raw.get('structured_slides') or []
    elif isinstance(slides_raw, list):
        slides = slides_raw
    else:
        slides = []
        
    normalized = []
    for s in slides:
        idx = s.get('slide_index') or s.get('index') or s.get('slide')
        try: idx = int(idx)
        except: idx = None
        normalized.append({'slide_index': idx, **s})
        
    for pos, s in enumerate(normalized, start=1):
        if s['slide_index'] is None:
            s['slide_index'] = pos
            
    normalized.sort(key=lambda x: int(x['slide_index']))
    return normalized

def normalize_whiteboards(wb_raw: Any) -> List[Dict[str,Any]]:
    if isinstance(wb_raw, dict):
        if 'results' in wb_raw and isinstance(wb_raw['results'], list):
            items = wb_raw['results']
        elif wb_raw.get('image_index') is not None:
            items = [wb_raw]
        else:
            items = wb_raw.get('images') or []
    elif isinstance(wb_raw, list):
        items = wb_raw
    else:
        items = []

    normalized = []
    for it in items:
        idx = it.get('image_index') if isinstance(it, dict) else None
        try: idx = int(idx) if idx is not None else None
        except: idx = None
        normalized.append({'image_index': idx, **(it if isinstance(it, dict) else {'raw': it})})
    return normalized

def build_wb_lookup(wb_list: List[Dict[str,Any]]) -> Dict[int, List[Dict[str,Any]]]:
    lookup = {}
    for wb in wb_list:
        idx = wb.get('image_index')
        if idx is not None:
            lookup.setdefault(int(idx), []).append(wb)
    return lookup

def find_nearest_slide_idx(target_idx: int, slide_indices: List[int]) -> Optional[int]:
    if not slide_indices: return None
    slide_indices_sorted = sorted(slide_indices)
    for s in reversed(slide_indices_sorted):
        if s <= target_idx: return s
    return slide_indices_sorted[0]

def merge_slides_and_whiteboards(slides: List[Dict], wbs: List[Dict], strategy: str = 'match_first') -> List[Dict]:
    slide_indices = [int(s['slide_index']) for s in slides]
    wb_lookup = build_wb_lookup(wbs)
    merged_sequence = []
    used_wb_ids = set()

    # Primary pass
    for s in slides:
        s_idx = int(s['slide_index'])
        merged_sequence.append({
            'type': 'slide',
            'index': s_idx,
            'content': s.get('structured_elements', s)
        })
        matched_wbs = wb_lookup.get(s_idx, [])
        for wb in matched_wbs:
            merged_sequence.append({
                'type': 'whiteboard',
                'index': wb.get('image_index'),
                'content': wb.get('structured_elements', wb)
            })
            used_wb_ids.add(id(wb))

    # Collect unmatched
    unmatched_wbs = [wb for wb in wbs if id(wb) not in used_wb_ids]

    if strategy == 'match_first':
        for wb in unmatched_wbs:
            target = wb.get('image_index')
            nearest = find_nearest_slide_idx(target if target is not None else math.inf, slide_indices)
            entry = {'type': 'whiteboard','index': wb.get('image_index'),'content': wb.get('structured_elements', wb)}
            if nearest is None:
                merged_sequence.append(entry)
            else:
                insert_pos = None
                for i, item in enumerate(merged_sequence):
                    if item['type'] == 'slide' and int(item['index']) == int(nearest):
                        insert_pos = i + 1
                if insert_pos is None: merged_sequence.append(entry)
                else: merged_sequence.insert(insert_pos, entry)
    
    return merged_sequence

def build_frames_from_merged(merged_sequence: List[Dict[str, Any]]) -> List[str]:
    frames = []
    for it in merged_sequence:
        typ = it.get('type', 'slide')
        idx = it.get('index', '?')
        header = f"\n<FRAME type='{typ}' index='{idx}'>\n"

        content_list = it.get('content', [])
        if isinstance(content_list, dict):
            content_list = [content_list]
        elif not isinstance(content_list, list):
            content_list = [content_list]

        slide_short_title = None
        for elem in content_list:
            if not isinstance(elem, dict):
                header += f"  <FRAME_TEXT>{str(elem)}</FRAME_TEXT>\n"
                continue

            e_type = (elem.get('type') or elem.get('content_type') or 'text').lower()
            content = (elem.get('content') or '').strip()
            explanation = (elem.get('explanation') or elem.get('rationale') or '').strip()
            img_path = elem.get('image_path', '') or elem.get('metadata', {}).get('image_path', '')

            if typ == 'slide' and not slide_short_title and e_type == 'text' and len(content) <= 120:
                slide_short_title = content.split('\n')[0].strip()

            if e_type in ('text', 'paragraph'):
                header += f"  <TEXT>{content}</TEXT>\n"
            elif e_type in ('equation', 'math'):
                img_ref = f" (Ref: {img_path})" if img_path else ""
                header += f"  <EQUATION>{content}{img_ref}</EQUATION>\n"
            elif e_type == 'table':
                header += f"  <TABLE_DATA>\n{content}\n</TABLE_DATA>\n"
            elif e_type == 'image':
                header += f"  <VISUAL_CONTEXT path='{img_path}'>Visual: {content}</VISUAL_CONTEXT>\n"
            else:
                header += f"  <FRAME_TEXT>{content}</FRAME_TEXT>\n"

            if explanation:
                header += f"  <INSTRUCTOR_ELABORATION>{explanation}</INSTRUCTOR_ELABORATION>\n"

        if slide_short_title:
            header += f"  <SLIDE_TITLE>{slide_short_title}</SLIDE_TITLE>\n"

        header += "</FRAME>"
        frames.append(header)
    return frames


# ==============================================================================
# --- MODAL GPU EXECUTION ---
# ==============================================================================

@app.cls(gpu="T4", image=image, scaledown_window=300, timeout=3600)
class CombinedNotesGeneratorGPU:
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
    def process(self, slides_json_str: str, wb_json_str: str, audio_json_str: str, supabase_url: str, supabase_key: str, deepseek_api_key: str, course_id: str):
        from nltk.tokenize import sent_tokenize
        from openai import OpenAI
        import markdown
        
        llm_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

        # 1. Parse Input
        slides_root = json.loads(slides_json_str)
        wb_root = json.loads(wb_json_str)
        audio_data = json.loads(audio_json_str)

        slides_list = normalize_slides(slides_root)
        wb_list = normalize_whiteboards(wb_root)

        # 2. RAG Helper
        def search_supabase(query: str, top_k: int = 3) -> List[Dict]:
            import torch
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

        # ==========================================================
        # PHASE 1: AUDIO ALIGNMENT (SLIDES ONLY)
        # ==========================================================
        print(f"\n🚀 PHASE 1: AUDIO ALIGNMENT (SLIDES ONLY)")
        for slide in slides_list:
            slide_index = slide.get('slide_index')
            audio_key = f"slide_{slide_index}"
            raw_audio = audio_data.get(audio_key, "")
            
            audio_sentences = [s.strip() for s in sent_tokenize(raw_audio) if s.strip()]
            if not audio_sentences: continue

            integrator = DynamicSlideIntegrator(self.embedder, threshold=0.45, dynamic_weight=0.20)
            integrator.extract_text_elements(slide)
            matches = integrator.align_dynamic(audio_sentences)

            for element in slide.get('structured_elements', []):
                e_id = element.get('id')
                if e_id in matches and matches[e_id]:
                    original_text = element.get('content', '')
                    matched_audio_text = " ".join(matches[e_id])
                    element['content'] = f"{original_text}\n\n{matched_audio_text}"

        # ==========================================================
        # PHASE 2: PARALLEL EXPLANATIONS (SLIDES & WHITEBOARDS)
        # ==========================================================
        print(f"\n🚀 PHASE 2: RAG & DEEPSEEK EXPLANATIONS")
        
        EXPLANATION_PROMPT = """You are an AI teaching assistant. 
Your task is to explain a specific text element from a course visual using the provided materials.

**Visual Context:**
{visual_context}

**Textbook/Background Knowledge:**
{rag_context}

**Text to Explain:**
"{target_text}"

**Instructions:**
1. Provide a direct, engaging explanation of the concept (2-3 sentences).
2. Synthesize the visual content with the detailed knowledge from the background context.
3. Keep the tone engaging and educational. Do not repeat the text verbatim.
4. **STRICTLY FORBIDDEN:** Do NOT use phrases like "The slide shows", "The textbook says", "The board mentions", or "According to the context". Speak directly to the student.

**Explanation:**"""

        # Explain Slides
        for slide in slides_list:
            for element in slide.get('structured_elements', []):
                if element.get('type') == 'text':
                    content = element.get('content', '')
                    slide_ctx = get_slide_context(slide, element.get('id'))
                    rag_ctx = format_rag_context(search_supabase(content, top_k=3))
                    user_prompt = EXPLANATION_PROMPT.format(target_text=content, visual_context=slide_ctx, rag_context=rag_ctx)
                    
                    try:
                        response = llm_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": user_prompt}],
                            max_tokens=300, temperature=0.4
                        )
                        element['explanation'] = response.choices[0].message.content.strip()
                    except Exception as e:
                        element['explanation'] = f"LLM Error: {str(e)}"
                    time.sleep(0.4)

        # Explain Whiteboards
        for board in wb_list:
            for element in board.get('structured_elements', []):
                if element.get('content_type') in ('text', 'paragraph'):
                    content = element.get('content', '')
                    if len(content) < 5: continue
                    board_ctx = get_board_context(board, element.get('original_id'))
                    rag_ctx = format_rag_context(search_supabase(content, top_k=3))
                    user_prompt = EXPLANATION_PROMPT.format(target_text=content, visual_context=board_ctx, rag_context=rag_ctx)
                    
                    try:
                        response = llm_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": user_prompt}],
                            max_tokens=300, temperature=0.4
                        )
                        element['explanation'] = response.choices[0].message.content.strip()
                    except Exception as e:
                        element['explanation'] = f"LLM Error: {str(e)}"
                    time.sleep(0.4)

        # ==========================================================
        # PHASE 3 & 4: MERGE & SUMMARIZE
        # ==========================================================
        print(f"\n🚀 PHASE 3 & 4: MERGING FRAMES & DEEPSEEK SUMMARIZATION")
        
        merged_sequence = merge_slides_and_whiteboards(slides_list, wb_list, strategy='match_first')
        frames = build_frames_from_merged(merged_sequence)

        BATCH_PROMPT = """
You are an expert Professor writing a textbook chapter.
**TASK:** Convert the raw slide and whiteboard data below into clean, seamless Markdown lecture notes.

**STRICT RULES:**
1. **NO CONVERSATION:** Do not write "Here are the notes", "In this slide", "Summary", or "End of batch". Output *only* the note content. Start directly with the first Header.
2. **Seamless Flow:** Write as if this is one continuous document. If the current frames continue the topic from the PREVIOUS NOTES, continue seamlessly WITHOUT repeating the main heading.
3. **Math:** Preserve ALL derivation steps and equations using LaTeX ($...$ or $$...$$). ALWAYS leave a blank line before and after block equations ($$...$$) for proper rendering.
4. **Detail:** Do not summarize. Include every rule, example, and definition.
5. **Images (CRITICAL):**
   - You will see tags like `<VISUAL_CONTEXT path='path/to/image.png'>Description</VISUAL_CONTEXT>`.
   - Insert the image using Markdown: `![Visual Description](path/to/image.png)`
   - Place the image immediately AFTER the text describing it.
   - ONLY include images that are **instructive** (e.g., apparatus, diagrams, specific examples like bacteria/molecules, charts). 
   - Discard: Do NOT include images described as "background", "abstract", "color", "shapes", or "decorative".
   - Captions: Do NOT write the visual description as text. Use it only for the Alt Text inside `![]`.
6. **Formatting Mastery (CRITICAL FOR LISTS):**
   - **Un-mash Lists:** Raw text often mashes bullet points onto a single line (e.g., `* Item 1 * Item 2 * Item 3`). YOU MUST format these into proper vertical Markdown lists, with each item on a new line.
   - **Blank Lines:** ALWAYS place an empty line before starting a list, table, or code block.
   - **Emphasis:** Use **bold** (`**text**`) for important terms, quantities, and definitions.
7. **Whiteboard Integration:** Merge whiteboard content smoothly into the relevant preceding slide's topic. Treat it as instructor elaboration, worked examples, or deep-dives. Do NOT create explicit "Whiteboard" headings unless it's a completely new topic.
8. **Context Awareness:** DO NOT repeat the text from the "PREVIOUS NOTES". It is only there so you know what was just discussed. Continue the document naturally.

**PREVIOUS NOTES (For context only - DO NOT REPEAT THIS):**
{previous_context}

**INPUT DATA (Frames {start}-{end}):**
{context_stream}
"""

        full_notes = ""
        last_batch_notes = ""
        BATCH_SIZE = 4
        total_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i : i + BATCH_SIZE]
            start_idx = i + 1
            end_idx = min(i + BATCH_SIZE, len(frames))
            print(f"⏳ Processing Batch { (i // BATCH_SIZE) + 1 }/{total_batches}...")
            
            batch_notes = process_batch(batch, start_idx, end_idx, last_batch_notes)
            
            # Cleanup leftover tags just in case
            batch_notes = re.sub(r"</?FRAME[^>]*>", "", batch_notes)
            
            full_notes += f"\n\n{batch_notes}"
            last_batch_notes = batch_notes
            time.sleep(1)

        full_notes = re.sub(r'\n{3,}', '\n\n', full_notes).strip()

        # ==========================================================
        # PHASE 5: HTML CONVERSION
        # ==========================================================
        print("\n🚀 PHASE 5: HTML CONVERSION")
        text = fix_list_spacing(full_notes)
        text = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})', lambda m: latex_table_to_html(m.group(0)), text, flags=re.DOTALL)

        math_blocks = {}
        def replace_math(match):
            # SAFE PLACEHOLDER: No underscores, asterisks, or spaces to prevent Markdown from altering it
            key = f"MATHBLOCKPLACEHOLDER{len(math_blocks)}END"
            math_blocks[key] = match.group(0)
            return key

        # 1. Protect $$...$$ (Display Math) FIRST so inner environments aren't orphaned
        text = re.sub(r'(\$\$.*?\$\$)', replace_math, text, flags=re.DOTALL)
        # 2. Protect block math environments (like \begin{align}...\end{align})
        text = re.sub(r'(\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\})', replace_math, text, flags=re.DOTALL)
        # 3. Protect $...$ (Inline Math)
        text = re.sub(r'(?<!\\)\$(?!\s)(?:\\.|[^$\\\n])+\$', replace_math, text)

        import markdown
        html_body = markdown.markdown(text, extensions=['extra', 'fenced_code', 'tables', 'sane_lists'])

        for key, value in math_blocks.items():
            html_body = html_body.replace(key, value)

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
            a:hover { text-decoration: underline; }
            hr { border: 0; border-top: 1px solid var(--border-color); margin: 2.5em 0; }
            blockquote { background-color: var(--gray-bg); border-left: 4px solid #4a4a4a; padding: 16px 20px; margin: 1.5em 0; border-radius: 0 4px 4px 0; color: #555; }
            figure { margin: 2em 0; text-align: center; }
            img { display: block; margin: 20px auto; max-width: 70%; max-height: 400px; width: auto; height: auto; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
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
          tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true, packages: {'[+]': ['noerrors', 'noundefined']} },
          options: { ignoreHtmlClass: 'tex2jax_ignore', processHtmlClass: 'tex2jax_process' },
          svg: { fontCache: 'global' }
        };
        </script>
        <script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        """

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Combined Lecture Notes</title>
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

    @web_app.post("/generate-combined-notes")
    async def generate_combined_notes(
        slides_json: str = Form(...),
        wb_json: str = Form(...),
        audio_json: str = Form(...),
        supabase_url: str = Form(...),
        supabase_key: str = Form(...),
        deepseek_api_key: str = Form(...),
        course_id: str = Form(...)
    ):
        processor = CombinedNotesGeneratorGPU()
        result = processor.process.remote(
            slides_json, wb_json, audio_json, 
            supabase_url, supabase_key, deepseek_api_key, course_id
        )
        return JSONResponse(content=result)

    return web_app