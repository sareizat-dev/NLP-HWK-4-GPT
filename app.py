# app.py — Chat con Ollama + RAG opcional (PDFs)
# ------------------------------------------------
# Variables útiles:
#   - OLLAMA_URL  (opcional) p.ej. "http://127.0.0.1:11435/api/chat"
#   - OLLAMA_PORT (opcional) por defecto "11434" si no se define OLLAMA_URL
#   - MODEL       (opcional) por defecto "llama3.1:8b-instruct-q4_0"

import os
import requests
import gradio as gr
from typing import List, Tuple

# --- RAG (LangChain + FAISS) ---
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embeddings wrapper correcto (evita el error de embed_documents)
EMB = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

VSTORE = None  # almacena el índice FAISS

def build_index(files: List[gr.File]) -> str:
    """Crea/actualiza el índice FAISS a partir de PDFs subidos."""
    global VSTORE
    if not files:
        return "Sube PDFs primero."

    # 1) Cargar documentos
    docs = []
    for f in files:
        try:
            loader = PyPDFLoader(f.name)
            docs.extend(loader.load())
        except Exception as e:
            return f"Error leyendo {getattr(f, 'name', 'archivo')}: {e}"

    # 2) Partir en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    # 3) Construir índice
    VSTORE = FAISS.from_texts(texts, embedding=EMB)
    return f"Índice RAG listo ✅  ({len(texts)} chunks)."

def retrieve(query: str, k: int = 3) -> str:
    """Recupera contexto relevante desde FAISS."""
    if not VSTORE:
        return ""
    # Compatibilidad entre versiones de LangChain
    if hasattr(VSTORE, "similarity_search_with_relevance_scores"):
        sims = VSTORE.similarity_search_with_relevance_scores(query, k=k)
        texts = [d.page_content for d, _ in sims]
    else:
        sims = VSTORE.similarity_search_with_score(query, k=k)
        texts = [d.page_content for d, _ in sims]
    return "\n\n".join(texts)

# --- Chat con Ollama ---
def _default_ollama_url() -> str:
    port = os.getenv("OLLAMA_PORT", "11434")
    return f"http://127.0.0.1:{port}/api/chat"

OLLAMA_URL = os.getenv("OLLAMA_URL", _default_ollama_url())
MODEL = os.getenv("MODEL", "llama3.1:8b-instruct-q4_0")

def chat_fn(message: str, history: List[Tuple[str, str]], use_rag: bool) -> str:
    """Llama a Ollama (/api/chat) y devuelve la respuesta."""
    messages = [{"role": "system", "content": "Eres un asistente técnico y conciso en español."}]

    for u, a in history:
        if u is None:
            continue
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    if use_rag:
        ctx = retrieve(message) or ""
        if ctx:
            message = (
                "Usa EXCLUSIVAMENTE el siguiente contexto para responder.\n\n"
                f"{ctx}\n\n"
                f"Pregunta del usuario: {message}"
            )

    messages.append({"role": "user", "content": message})

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception as e:
        return f"⚠️ Error llamando a Ollama en {OLLAMA_URL}: {e}"

# --- UI (Gradio) ---
with gr.Blocks() as demo:
    gr.Markdown("# Chat LLM (Ollama) + RAG opcional")
    gr.Markdown(
        f"**Modelo:** `{MODEL}`  —  **Endpoint:** `{OLLAMA_URL}`  \n"
        "Activa **Usar RAG** para que el modelo responda usando PDF(s) que subas."
    )

    with gr.Row():
        use_rag = gr.Checkbox(label="Usar RAG", value=False)
        pdfs = gr.File(file_count="multiple", file_types=[".pdf"], label="PDFs para RAG")
        build_btn = gr.Button("Construir índice")
        status = gr.Markdown()

    build_btn.click(build_index, inputs=[pdfs], outputs=[status])

    chat = gr.Chatbot(height=380)
    msg = gr.Textbox(placeholder="Escribe aquí…")
    send = gr.Button("Enviar", variant="primary")

    def _push_user(m, h):
        if not m:
            return "", h
        h = h + [(m, None)]
        return "", h

    def _answer(h, rag):
        if not h:
            return h
        q = h[-1][0]
        a = chat_fn(q, h[:-1], rag)
        h[-1] = (q, a)
        return h

    msg.submit(_push_user, [msg, chat], [msg, chat]).then(_answer, [chat, use_rag], [chat])
    send.click(_push_user, [msg, chat], [msg, chat]).then(_answer, [chat, use_rag], [chat])

demo.launch(server_name="0.0.0.0", server_port=7860, share = True)

