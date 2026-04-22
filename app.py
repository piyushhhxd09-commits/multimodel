import os
import re
import io
import time
import base64
import traceback
import pickle
import concurrent.futures
import difflib
from typing import List, Dict

import fitz  # PyMuPDF
import pdfplumber
import gradio as gr
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient

import db  # SQLite Logging Module

# =============================================================
# CONFIGURATION
# =============================================================
HF_MODEL_GEN = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_CAPTION = "Salesforce/blip-image-captioning-base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_CHUNKS = 12
RERANK_TOP_N = 8

_models = {}

def get_embed_model():
    if "embed" not in _models:
        print("[INIT] Loading Embedding Model ...")
        _models["embed"] = SentenceTransformer(EMBED_MODEL_NAME)
    return _models["embed"]

def get_rerank_model():
    if "rerank" not in _models:
        print("[INIT] Loading Reranker Model ...")
        _models["rerank"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _models["rerank"]

def get_hf_client(model: str = HF_MODEL_GEN):
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        return None
    return InferenceClient(model=model, token=token.strip())

# =============================================================
# UTILS
# =============================================================
def clean_text(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def table_to_markdown(table: list) -> str:
    if not table or len(table) < 1:
        return ""
    cleaned = [[str(c).replace("\n", " ").strip() if c else "" for c in row] for row in table]
    if all(all(c == "" for c in r) for r in cleaned):
        return ""
    header = cleaned[0]
    ncols = len(header)
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * ncols) + " |\n"
    for row in cleaned[1:]:
        while len(row) < ncols:
            row.append("")
        md += "| " + " | ".join(row[:ncols]) + " |\n"
    return md.strip()

def pil_to_base64(img: Image.Image, max_px: int = 800) -> str:
    buf = io.BytesIO()
    img = img.copy()
    img.thumbnail((max_px, max_px))
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

# =============================================================
# EXTRACTION
# =============================================================
def extract_pdf_comprehensive(path: str, progress=gr.Progress()):
    all_chunks: List[Dict] = []
    images: List[Dict] = []

    progress(0.1, desc="Opening PDF...")
    doc = fitz.open(path)
    current_concept = "Intro/Overview"

    total_pages = len(doc)
    for pn in range(total_pages):
        progress((pn + 1) / total_pages * 0.7, desc=f"Processing Page {pn+1}/{total_pages}...")
        page = doc[pn]

        # 1. TABLE EXTRACTION (fitz)
        try:
            tabs = page.find_tables()
            for tab in tabs:
                md = table_to_markdown(tab.extract())
                if len(md) > 20:
                    all_chunks.append({
                        "page": pn + 1,
                        "text": f"### [TABLE DATA]\n{md}",
                        "type": "table",
                        "concept": current_concept,
                    })
        except Exception:
            pass

        # 2. TEXT EXTRACTION
        try:
            blocks = page.get_text("dict", flags=11)["blocks"]
        except Exception:
            blocks = []
        page_text_acc = []
        for b in blocks:
            if b.get("type") == 0:
                block_text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        txt = clean_text(span.get("text", ""))
                        if not txt:
                            continue
                        if ("bold" in span.get("font", "").lower() or span.get("size", 0) > 12) and len(txt) < 100:
                            current_concept = txt
                        block_text += txt + " "
                if len(block_text.strip()) > 15:
                    page_text_acc.append(block_text.strip())

        full_page_text = " ".join(page_text_acc)
        words = full_page_text.split()
        for i in range(0, len(words), 350):
            chunk_words = words[i: i + 450]
            if len(chunk_words) < 10:
                continue
            all_chunks.append({
                "page": pn + 1,
                "text": " ".join(chunk_words),
                "type": "text",
                "concept": current_concept,
            })

        # 3. IMAGE EXTRACTION
        try:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                img_data = base_img["image"]
                try:
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                except Exception:
                    continue
                if pil_img.width < 50 or pil_img.height < 50:
                    continue
                if np.array(pil_img).std() < 2:
                    continue

                rects = page.get_image_rects(xref)
                surrounding_text = ""
                if rects:
                    search_rect = rects[0] + (-50, -50, 50, 200)
                    surrounding_text = page.get_text("text", clip=search_rect).strip()

                images.append({
                    "page": pn + 1,
                    "base64": pil_to_base64(pil_img),
                    "caption": f"Diagram: {surrounding_text[:1000]}",
                    "concept": current_concept,
                    "image": pil_img,
                })
        except Exception:
            pass

    # Fallback Table Extraction (pdfplumber)
    progress(0.8, desc="Finalizing tables...")
    try:
        with pdfplumber.open(path) as pdf:
            for idx, p in enumerate(pdf.pages):
                for tbl in p.extract_tables() or []:
                    md = table_to_markdown(tbl)
                    if len(md) > 30 and not any(md[:50] in c["text"] for c in all_chunks if c["type"] == "table"):
                        all_chunks.append({
                            "page": idx + 1,
                            "text": f"### [TABLE DATA]\n{md}",
                            "type": "table",
                            "concept": "Appendix",
                        })
    except Exception:
        pass

    # Parallel AI Captioning
    def process_image(img_info):
        try:
            client = get_hf_client(HF_MODEL_CAPTION)
            if client is None:
                return img_info
            buf = io.BytesIO()
            img_info['image'].save(buf, format="JPEG")
            res = client.image_to_text(buf.getvalue())
            cap = res.generated_text if hasattr(res, "generated_text") else str(res)
            cap = cap.strip()
            if cap:
                img_info['caption'] = f"{img_info['caption']} | AI View: {cap}"
        except Exception:
            pass
        return img_info

    if images:
        progress(0.9, desc="AI Captioning technical diagrams...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            images = list(executor.map(process_image, images))

        # Sync searchable chunks with final captions
        for im in images:
            all_chunks.append({
                "page": im['page'],
                "text": f"### [TECHNICAL DIAGRAM]\n{im['caption']}",
                "type": "image_meta",
                "concept": im['concept'],
            })
            # Drop PIL to save memory
            im.pop('image', None)

    doc.close()
    progress(1.0, desc="Extraction Complete ✓")
    return all_chunks, images

# =============================================================
# HYBRID RETRIEVAL
# =============================================================
def hybrid_retrieve(query: str, chunks: List[Dict], embs: np.ndarray, bm25: BM25Okapi, top_k: int = 15):
    if not chunks or embs is None:
        return []
    model = get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    vs = np.dot(embs, q_emb.T).squeeze()
    if vs.ndim == 0:
        vs = np.array([float(vs)])

    stop_words = {"what", "is", "the", "a", "an", "of", "and", "in", "to", "for", "with", "on", "by", "at"}
    query_tokens = [w for w in query.lower().split() if w not in stop_words]
    bs = np.array(bm25.get_scores(query_tokens), dtype="float32")

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9) if x.max() > x.min() else np.zeros_like(x)

    scores = 0.6 * normalize(vs) + 0.4 * normalize(bs)
    top_indices = np.argsort(scores)[::-1][:top_k]
    candidate_chunks = [chunks[int(i)] for i in top_indices]

    reranker = get_rerank_model()
    pairs = [[query, c["text"][:600]] for c in candidate_chunks]
    rerank_scores = reranker.predict(pairs)
    final_indices = np.argsort(rerank_scores)[::-1]
    sorted_chunks = [candidate_chunks[int(i)] for i in final_indices]

    results = []
    for i, chunk in enumerate(sorted_chunks):
        if i < 6 or rerank_scores[final_indices[i]] > -2.0:
            results.append(chunk)
    return results if results else [sorted_chunks[0]]

# =============================================================
# CHAT HANDLER
# =============================================================
def chat_handler(query, history, chunks, images, embs, bm25):
    if history is None:
        history = []
    if not chunks or embs is None:
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a PDF document first."})
        return history, "", []

    try:
        # Spelling correction
        doc_vocab = set()
        for c in chunks:
            doc_vocab.add(c['concept'].lower())
            for word in re.findall(r'\b[a-zA-Z]{3,}\b', c['text'].lower()):
                doc_vocab.add(word)

        corrected_words = []
        for word in query.lower().split():
            matches = difflib.get_close_matches(word, list(doc_vocab), n=1, cutoff=0.8)
            corrected_words.append(matches[0] if matches else word)
        search_query = " ".join(corrected_words)

        query_lower = search_query.lower()
        query_words_set = set(query_lower.split())

        visual_keywords = {"picture", "image", "diagram", "figure", "photo", "show", "visual", "pic", "draw", "schematic", "map", "chart", "graph"}
        question_keywords = {"what", "how", "where", "why", "when", "who", "which", "explain", "advantages", "disadvantages", "adv", "define", "describe", "difference", "compare"}

        is_stage1_visual = any(vk in query_words_set for vk in visual_keywords)
        is_stage2_question = any(qk in query_words_set for qk in question_keywords) or "?" in query_lower

        if is_stage1_visual:
            mode = "STAGE1_VISUAL"
            max_context = 7
            system_prompt = (
                "You are a strict technical assistant. You must ONLY use the provided CONTEXT.\n"
                "The user is asking for a specific visual (image, diagram, picture, etc.).\n"
                "1. Provide a very brief 1-2 sentence description of the requested visual based on the context.\n"
                "2. If the visual or topic is not in the context, say 'No info in document.'\n"
                "3. Cite with [Page X].\n"
                "4. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION."
            )
        elif is_stage2_question:
            mode = "STAGE2_QUESTION"
            max_context = 7
            system_prompt = (
                "You are a strict technical assistant. You must ONLY use the provided CONTEXT.\n"
                "The user is asking a specific question.\n"
                "1. Answer ONLY what is specifically asked. Be concise and specific. Do NOT add extra unasked info.\n"
                "2. If the answer is not explicitly in the context, say 'No info in document.'\n"
                "3. Cite with [Page X].\n"
                "4. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION."
            )
        else:
            mode = "STAGE3_GENERAL"
            max_context = 15
            system_prompt = (
                "You are a strict technical assistant. You must ONLY use the provided CONTEXT.\n"
                "The user is asking about a general topic.\n"
                "1. Extract and summarize EVERYTHING related to this topic from the context.\n"
                "2. If the topic is not in the context, say 'No info in document.'\n"
                "3. Use bullet points for readability.\n"
                "4. Cite with [Page X].\n"
                "5. NO OUTSIDE KNOWLEDGE. NO HALLUCINATION."
            )

        top_chunks = hybrid_retrieve(search_query, chunks, embs, bm25, top_k=20)
        reranker = get_rerank_model()
        text_pairs = [[search_query, c['text'][:600]] for c in top_chunks]
        text_scores = reranker.predict(text_pairs)
        scored_chunks = sorted(zip(top_chunks, text_scores), key=lambda x: x[1], reverse=True)
        final_top_chunks = [c for c, _ in scored_chunks[:max_context]]

        context_parts = [f"--- [SOURCE: Page {c['page']} | Section: {c['concept']}] ---\n{c['text']}" for c in final_top_chunks]
        context_str = "\n\n".join(context_parts)[:15000]

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-6:]:
            content = str(turn.get("content", ""))
            for marker in ["### \U0001f4ca", "### \U0001f5bc", "<div", "Data Evidence", "Visual Evidence"]:
                if marker in content:
                    content = content.split(marker)[0]
            messages.append({"role": turn["role"], "content": content.strip()})
        messages.append({"role": "user", "content": f"CONTEXT:\n{context_str}\n\nUSER QUESTION: {query}"})

        client = get_hf_client()
        if not client:
            ans = "⚠️ **API Token Missing.** Please set `HF_TOKEN` in secrets."
        else:
            try:
                response = client.chat_completion(messages=messages, max_tokens=1000, temperature=0.0)
                ans = response.choices[0].message.content.strip()
            except Exception as e:
                ans = f"❌ **Cloud Error:** {str(e)}"

        # Visuals / Tables per stage
        final_rel_imgs = []
        final_rel_tabs = []

        if "no info in document" not in ans.lower():
            if mode == "STAGE1_VISUAL" and images:
                img_pairs = [[search_query, im['caption']] for im in images]
                img_scores = reranker.predict(img_pairs)
                scored_imgs = sorted(zip(images, img_scores), key=lambda x: x[1], reverse=True)
                if scored_imgs and scored_imgs[0][1] > 1.7:
                    final_rel_imgs = [scored_imgs[0][0]]
            elif mode == "STAGE3_GENERAL":
                if images:
                    img_pairs = [[search_query, im['caption']] for im in images]
                    img_scores = reranker.predict(img_pairs)
                    scored_imgs = sorted(zip(images, img_scores), key=lambda x: x[1], reverse=True)
                    for im, score in scored_imgs:
                        if score > 1.7:
                            final_rel_imgs.append(im)
                rel_tabs = [c for c in chunks if c['type'] == 'table']
                if rel_tabs:
                    tab_pairs = [[search_query, t['text'][:500]] for t in rel_tabs]
                    tab_scores = reranker.predict(tab_pairs)
                    scored_tabs = sorted(zip(rel_tabs, tab_scores), key=lambda x: x[1], reverse=True)
                    for t, score in scored_tabs:
                        if score > 0.0:
                            final_rel_tabs.append(t)
            # STAGE2_QUESTION: text-only, no visuals/tables

        tables_md = ""
        if final_rel_tabs:
            tables_md = "\n\n### \U0001f4ca Data Evidence\n"
            for t in final_rel_tabs:
                tables_md += f"<div style='background:#1e293b;padding:15px;border-radius:12px;border:1px solid #334155;margin-bottom:12px;'>\n\n{t['text']}\n\n*(Source: Page {t['page']})*</div>\n\n"

        image_md = ""
        if final_rel_imgs:
            image_md += "\n\n### \U0001f5bc Visual Evidence\n<div style='display:flex;gap:15px;overflow-x:auto;padding-bottom:10px;'>"
            for im in final_rel_imgs:
                dc = im['caption'][:120] + "..." if len(im['caption']) > 120 else im['caption']
                image_md += f"<div style='flex:0 0 400px;background:#1e293b;padding:10px;border-radius:12px;border:1px solid #334155;'><img src='data:image/jpeg;base64,{im['base64']}' style='width:100%;border-radius:8px;'><br><p style='font-size:12px;margin-top:8px;color:#94a3b8;'>Page {im['page']} | {dc}</p></div>"
            image_md += "</div>"

        final_answer = ans + tables_md + image_md
        try:
            db.log_chat(query, final_answer, "PDF Document")
        except Exception:
            pass

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": final_answer})
        return history, "", final_rel_imgs

    except Exception as e:
        traceback.print_exc()
        err = str(e)
        if "402" in err:
            msg = "❌ **Hugging Face Quota Exceeded.**"
        elif "429" in err:
            msg = "⚠️ **Rate Limit.** Try again in a moment."
        else:
            msg = f"❌ **System Error:** {err}"
        history.append({"role": "assistant", "content": msg})
        return history, "", []

# =============================================================
# UI
# =============================================================
CSS = """
body { background: #212121; color: #ececf1; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
#sidebar { background: #171717; border-right: 1px solid #343541; padding: 20px; min-height: 100vh; }
#main-chat { background: #212121; padding: 20px; }
#chatbot { background: transparent !important; border: none !important; }
#input-container { background: #2f2f2f; border-radius: 12px; border: 1px solid #4d4d4f; padding: 10px; margin-top: 20px; }
#query-box textarea { background: transparent !important; border: none !important; color: white !important; }
#send-btn { background: #676767 !important; border-radius: 8px !important; color: #212121 !important; }
.sidebar-label { color: #8e8ea0; font-size: 11px; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; }
"""

def process_and_init(file_obj, progress=gr.Progress()):
    if not file_obj:
        return "⚠️ No file uploaded", [], [], None, None, "⚠️ Pending", []
    try:
        chunks, images = extract_pdf_comprehensive(file_obj.name, progress)
        progress(0.9, desc="Finalizing index...")
        model = get_embed_model()
        texts = [c["text"] for c in chunks]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        try:
            with open("engine_cache.pkl", "wb") as f:
                pickle.dump({"chunks": chunks, "images": images, "embs": embs, "bm25": bm25}, f)
        except Exception:
            pass
        return ("✅ Engine Ready", chunks, images, embs, bm25, "✔️ Active",
                [{"role": "assistant", "content": "✅ **System Initialized!** Ask about text, tables, or diagrams."}])
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error: {str(e)}", [], [], None, None, "❌ Error", [{"role": "assistant", "content": f"❌ Error: {str(e)}"}]

def load_cached_engine():
    if os.path.exists("engine_cache.pkl"):
        try:
            with open("engine_cache.pkl", "rb") as f:
                d = pickle.load(f)
            return ("✅ Engine Ready (Cached)", d["chunks"], d["images"], d["embs"], d["bm25"], "✔️ Active",
                    [{"role": "assistant", "content": "👋 **Welcome back!** Ready to analyze your document."}])
        except Exception:
            pass
    return "📤 Upload PDF", [], [], None, None, "⚠️ Pending", [{"role": "assistant", "content": "👋 **Hello!** Upload a PDF to begin."}]

with gr.Blocks(css=CSS, title="Multimodal PDF Assistant") as demo:
    st_chunks, st_images, st_embs, st_bm25 = gr.State([]), gr.State([]), gr.State(None), gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar", min_width=280):
            new_chat_btn = gr.Button("➕ New Chat", variant="secondary")
            gr.Markdown("### 📄 Document", elem_classes="sidebar-label")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], container=False)
            process_btn = gr.Button("🚀 Process PDF", variant="primary")
            status_msg = gr.Markdown("Ready.")
            status_badge = gr.Markdown("")

            with gr.Accordion("📖 GUIDE", open=False):
                gr.Markdown(
                    "- **Q&A (Stage 2)**: ask 'what/how/why' for precise facts.\n"
                    "- **Visual (Stage 1)**: mention 'image/diagram/figure' to retrieve a visual.\n"
                    "- **Concept (Stage 3)**: type a topic for full extraction."
                )

        with gr.Column(scale=4, elem_id="main-chat"):
            # NOTE: type="messages" removed for compatibility with older Gradio.
            # The history dict format {"role","content"} still works in classic mode.
            chatbot = gr.Chatbot(show_label=False, elem_id="chatbot", height=750, type="messages") \
                if "type" in gr.Chatbot.__init__.__code__.co_varnames else gr.Chatbot(show_label=False, elem_id="chatbot", height=750)
            with gr.Row(elem_id="input-container"):
                query_box = gr.Textbox(placeholder="Message PDF Assistant...", scale=10, container=False, elem_id="query-box")
                send_btn = gr.Button("▲", scale=1, elem_id="send-btn")

    new_chat_btn.click(lambda: ([], ""), None, [chatbot, query_box])
    demo.load(load_cached_engine, None, [status_msg, st_chunks, st_images, st_embs, st_bm25, status_badge, chatbot])
    process_btn.click(process_and_init, [pdf_input], [status_msg, st_chunks, st_images, st_embs, st_bm25, status_badge, chatbot])

    def run_chat_flow(q, h, chunks, imgs, embs, bm25):
        if not q or not q.strip():
            return h, ""
        new_h, _, _ = chat_handler(q, h, chunks, imgs, embs, bm25)
        return new_h, ""

    send_btn.click(run_chat_flow, [query_box, chatbot, st_chunks, st_images, st_embs, st_bm25], [chatbot, query_box])
    query_box.submit(run_chat_flow, [query_box, chatbot, st_chunks, st_images, st_embs, st_bm25], [chatbot, query_box])

if __name__ == "__main__":
    db.init_db()
    demo.launch(server_name="0.0.0.0", server_port=7860)
