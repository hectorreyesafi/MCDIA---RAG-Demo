# MCDIA — RAG Demo

Aplicación Streamlit de preguntas y respuestas sobre documentos PDF usando una arquitectura **RAG** (*Retrieval-Augmented Generation*) con **OpenAI** y **Qdrant** como base de datos vectorial en memoria.

## Funcionalidades

- 📤 Sube uno o varios archivos PDF
- ✂️ Extrae y divide el texto en fragmentos manejables
- 🔢 Genera embeddings con `text-embedding-3-small` (OpenAI)
- 🗄️ Almacena los vectores en Qdrant (en memoria)
- 💬 Introduce una pregunta en lenguaje natural
- 🔍 Recupera los fragmentos más relevantes por similitud coseno
- 🤖 Genera una respuesta con `gpt-4o-mini` (OpenAI) usando el contexto recuperado
- 📚 Muestra los fragmentos recuperados como contexto

## Requisitos

- Python ≥ 3.10
- Clave de API de OpenAI

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` a partir del ejemplo:

```bash
cp .env.example .env
```

Edita `.env` y añade tu clave de OpenAI:

```
OPENAI_API_KEY=sk-...tu-clave...
```

## Ejecución

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`.

## Arquitectura

```
PDF ──► Extracción de texto ──► Fragmentación ──► Embeddings (OpenAI)
                                                          │
                                                    Qdrant (memoria)
                                                          │
Pregunta ──► Embedding (OpenAI) ──► Búsqueda por similitud
                                          │
                                    Fragmentos relevantes
                                          │
                              GPT-4o-mini (OpenAI) ──► Respuesta
```