"""
RAG Demo — Preguntas sobre documentos PDF
Aplicación Streamlit con OpenAI + Qdrant (en memoria)
"""

import io
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
COLLECTION_NAME = "rag_demo"
EMBEDDING_DIM = 1536        # dimensión de text-embedding-3-small
CHUNK_SIZE = 500            # caracteres por fragmento
CHUNK_OVERLAP = 50          # solapamiento entre fragmentos
TOP_K = 5                   # fragmentos más relevantes a recuperar

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae todo el texto de un archivo PDF."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en fragmentos con solapamiento."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_openai_client() -> OpenAI:
    """Crea y devuelve el cliente de OpenAI."""
    api_key = st.session_state.get("openai_api_key") or OPENAI_API_KEY
    if not api_key:
        st.error("🔑 No se encontró la clave de OpenAI. Configura OPENAI_API_KEY o introdúcela en la barra lateral.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos usando OpenAI."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_qdrant_client() -> QdrantClient:
    """Devuelve el cliente Qdrant en memoria (singleton por sesión)."""
    if "qdrant_client" not in st.session_state:
        st.session_state.qdrant_client = QdrantClient(":memory:")
    return st.session_state.qdrant_client


def ensure_collection(qdrant: QdrantClient) -> None:
    """Crea la colección Qdrant si no existe."""
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def index_chunks(qdrant: QdrantClient, client: OpenAI, chunks: list[str], source: str) -> int:
    """Genera embeddings e indexa los fragmentos en Qdrant. Devuelve el número de puntos insertados."""
    embeddings = get_embeddings(client, chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "source": source},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search_similar(qdrant: QdrantClient, client: OpenAI, query: str, top_k: int = TOP_K) -> list[dict]:
    """Busca los fragmentos más similares a la consulta."""
    query_embedding = get_embeddings(client, [query])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return [{"text": r.payload["text"], "source": r.payload["source"], "score": r.score} for r in response.points]


def generate_answer(client: OpenAI, question: str, context_chunks: list[dict]) -> str:
    """Genera una respuesta en lenguaje natural usando el contexto recuperado."""
    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in context_chunks
    )
    system_prompt = (
        "Eres un asistente experto en responder preguntas basándote únicamente en el contexto "
        "proporcionado. Si la respuesta no se encuentra en el contexto, indícalo claramente. "
        "Responde siempre en el mismo idioma que la pregunta."
    )
    user_prompt = f"Contexto:\n{context}\n\nPregunta: {question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Demo — Preguntas sobre PDFs",
    page_icon="📄",
    layout="centered",
)

st.title("📄 RAG Demo — Preguntas sobre documentos PDF")
st.markdown(
    "Sube uno o varios archivos PDF, escribe una pregunta y obtén una respuesta "
    "generada con **OpenAI** a partir de los fragmentos más relevantes de tus documentos."
)

# ── Barra lateral ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input(
        "Clave de OpenAI",
        type="password",
        value=st.session_state.get("openai_api_key", OPENAI_API_KEY),
        help="Introduce tu clave de API de OpenAI. También puedes definirla en la variable de entorno OPENAI_API_KEY.",
    )
    if api_key_input:
        st.session_state["openai_api_key"] = api_key_input

    st.markdown("---")
    st.markdown(
        "**Modelos utilizados**\n"
        f"- Embeddings: `{EMBEDDING_MODEL}`\n"
        f"- Chat: `{CHAT_MODEL}`"
    )
    st.markdown("---")
    if st.button("🗑️ Limpiar base de datos vectorial"):
        for key in ("qdrant_client", "indexed_files"):
            st.session_state.pop(key, None)
        st.success("Base de datos vaciada.")

# ── Sección 1: Subir PDFs ──────────────────────────────────────────────────
st.header("1️⃣ Sube tus documentos PDF")
uploaded_files = st.file_uploader(
    "Selecciona uno o varios archivos PDF",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("⚙️ Procesar PDFs e indexar en Qdrant"):
        qdrant = get_qdrant_client()
        ensure_collection(qdrant)
        openai_client = get_openai_client()

        indexed_files = st.session_state.get("indexed_files", set())
        new_files = [f for f in uploaded_files if f.name not in indexed_files]

        if not new_files:
            st.info("Todos los archivos ya han sido indexados.")
        else:
            progress_bar = st.progress(0, text="Procesando PDFs…")
            total = len(new_files)
            for i, pdf_file in enumerate(new_files):
                with st.spinner(f"Procesando {pdf_file.name}…"):
                    raw_text = extract_text_from_pdf(pdf_file.read())
                    if not raw_text.strip():
                        st.warning(f"⚠️ No se pudo extraer texto de **{pdf_file.name}**. Puede ser un PDF de imágenes.")
                        continue
                    chunks = split_text(raw_text)
                    n = index_chunks(qdrant, openai_client, chunks, source=pdf_file.name)
                    indexed_files.add(pdf_file.name)
                    st.success(f"✅ **{pdf_file.name}** — {n} fragmentos indexados.")
                progress_bar.progress((i + 1) / total, text=f"Procesando PDFs… ({i+1}/{total})")

            st.session_state["indexed_files"] = indexed_files
            progress_bar.empty()

# ── Sección 2: Hacer una pregunta ─────────────────────────────────────────
st.header("2️⃣ Haz una pregunta")
question = st.text_input(
    "Escribe tu pregunta aquí",
    placeholder="¿De qué trata el documento? ¿Cuáles son los puntos principales?",
)

ask_button = st.button("🔍 Consultar", disabled=not question.strip())

if ask_button and question.strip():
    qdrant = get_qdrant_client()
    openai_client = get_openai_client()

    collections = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME not in collections:
        st.warning("⚠️ No hay documentos indexados. Por favor, sube y procesa primero tus PDFs.")
    else:
        with st.spinner("Buscando fragmentos relevantes…"):
            fragments = search_similar(qdrant, openai_client, question)

        if not fragments:
            st.warning("No se encontraron fragmentos relevantes. Intenta con otra pregunta.")
        else:
            with st.spinner("Generando respuesta…"):
                answer = generate_answer(openai_client, question, fragments)

            # ── Respuesta generada ────────────────────────────────────────
            st.header("💬 Respuesta")
            st.markdown(answer)

            # ── Fragmentos recuperados ────────────────────────────────────
            st.header("📚 Fragmentos recuperados como contexto")
            for idx, frag in enumerate(fragments, start=1):
                with st.expander(
                    f"Fragmento {idx} — {frag['source']} (similitud: {frag['score']:.3f})"
                ):
                    st.write(frag["text"])
