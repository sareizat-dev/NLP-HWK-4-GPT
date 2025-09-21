# Tarea 4: Despliegue LLM en Lightning (Ollama + RAG)

**Nota**: Se grabó un video mostrando el flujo de la solución (chat normal, carga de PDFs, chat con RAG y la plantilla de Lightning). No se incluyen Public URLs para evitar consumo de creditos. El video se encuentra en la carpeta de Google Drive de tarea 4, en el siguiente enlace:https://drive.google.com/file/d/1yMiyiY2YsjH1Y9VEvwfMaNDU1XFKdAja/view?usp=drive_link 

---

## Resumen de la solución
Este repositorio contiene:
- App propia con interfaz de chat (Gradio) conectada a Ollama y RAG básico sobre PDFs.
- Ejecucion de la plantilla oficial de Lightning "Chat with Meta's Llama 3 8B" como segundo chat independiente.

---

## Objetivos de la entrega
1. Desplegar un modelo abierto con Ollama y exponer una UI tipo chat.
2. Incluir una funcionalidad extra: RAG (busqueda semantica en PDFs + inyeccion de contexto).
3. Levantar la plantilla oficial de Lightning para comparar enfoques.

---

## Arquitectura (alto nivel)

Componentes:
- Ollama (servidor local): modelo `llama3.1:8b-instruct-q4_0` por REST (`/api/chat`).
- App de usuario (Gradio): chat + carga de PDFs -> embeddings `all-MiniLM-L6-v2` -> FAISS -> retrieval opcional.
- Lightning Platform (Studio): host GPU (T4 16 GB) para app y servidor de modelo.
- Plantilla Lightning utilizada: "Chat with Meta's Llama 3 8B" (https://lightning.ai/lightning-ai/studios/chat-with-meta-s-llama-3-8b?view=public&section=featured&query=chat+with+llama).

Flujo de la funcionalidad RAG (cuando se activa):
1) Subir PDFs y dividir en chunks (800/120).
2) Generar embeddings con `sentence-transformers/all-MiniLM-L6-v2`.
3) Indexar en FAISS (en memoria).
4) Recuperar k chunks relevantes por pregunta e inyectarlos como contexto al prompt.

---

## Entorno y dependencias

Plataforma:
- Lightning Platform (GPU T4 16 GB).

Modelos:
- App propia: `llama3.1:8b-instruct-q4_0` (Ollama).
- Plantilla: "Chat with Meta's Llama 3 8B" (Lightning Template).

Stack:
- Python 3.x, gradio, requests, langchain, langchain-community, faiss-cpu, sentence-transformers, pypdf.

Instalacion:
```bash
pip install -r requirements.txt
---

## Ejecución Local
#1) Arrancar Ollama y descargar el modelo
ollama serve &
ollama pull llama3.1:8b-instruct-q4_0

# 2) Exportar el endpoint si no es el default
export OLLAMA_URL="http://127.0.0.1:11434/api/chat"

# 3) Levantar la app
python app.py



