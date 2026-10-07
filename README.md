
# Sistema RAG escalable con Pinecone

Módulo de recuperación híbrida (BM25 + búsqueda vectorial) sobre la documentación del tutorial de FastAPI, con índice en Pinecone Serverless y evaluación con Precision@5 y Recall@5.

Pre-entrega 4 — IA Engineering.

## Componentes

| Archivo | Qué hace |
|---|---|
| `config.py` | Configuración compartida (claves, índice, namespace, modelo, dimensión) |
| `setup_index.py` | Verifica si el índice existe y lo crea en modo Serverless si hace falta |
| `ingest.py` | Carga los documentos, los divide en chunks, genera embeddings y los sube a Pinecone |
| `rag_system.py` | Clase `RAGSystem`: combina BM25 y Pinecone con un `EnsembleRetriever` y devuelve el top-5 |
| `evaluate.py` | Calcula Recall@5 y Precision@5 sobre el golden set y compara BM25, vectorial e híbrido |
| `golden_set.json` | 5 preguntas con el documento fuente esperado |
| `data/docs/` | Dataset: 8 páginas del tutorial oficial de FastAPI (Markdown) |

## Decisiones de diseño

- **Dataset:** 8 páginas del tutorial de FastAPI (`handling-errors`, `path-params`, `query-params`, `body`, `cors`, `background-tasks`, `middleware`, `dependencies`), elegidas por cubrir temas bien diferenciados.
- **Embeddings locales y gratuitos:** `sentence-transformers/all-MiniLM-L6-v2`, dimensión **384**. El índice se crea con esa misma dimensión, tomada de `config.py`, para evitar el mismatch de dimensiones. Por eso `OPENAI_API_KEY` está en el `.env` pero es opcional.
- **Idioma:** el modelo y la documentación están en inglés, así que las preguntas del golden set también.
- **Chunking:** `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `chunk_size=600` y `chunk_overlap=100` tokens (dentro del rango 500-800 sugerido). Resultado: 33 chunks.
- **Metadatos:** cada chunk guarda `source` (archivo), `category` (etiqueta temática) y `chunk_index`. El texto original se guarda en `metadata["text"]`, así una consulta a Pinecone trae vector + texto + fuente sin una base de datos adicional.
- **Namespace:** todos los vectores van al namespace `fastapi-docs`.
- **IDs fijos** (`archivo-chunk_index`): si se vuelve a correr la ingesta, los vectores se sobrescriben en lugar de duplicarse.
- **BM25:** corre en memoria, así que la ingesta guarda los chunks en `chunks.json` para reconstruirlo. Usa un tokenizador que pasa a minúsculas e ignora la puntuación, para que términos como `HTTPException` coincidan aunque aparezcan seguidos de un signo.
- **Ensemble:** pesos 0.5 / 0.5 entre BM25 y vectorial.

## Cómo replicar el índice

### 1. Requisitos

- Python 3.10 o superior
- Cuenta gratuita en [Pinecone](https://app.pinecone.io) y su API key (sección **API Keys**)

### 2. Instalación

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash en Windows
# .venv\Scripts\activate        # PowerShell / CMD
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
```

### 3. Variables de entorno

Copiar `.env.example` como `.env` y completar:

```
PINECONE_API_KEY=tu_clave
OPENAI_API_KEY=
INDEX_NAME=fastapi-docs-rag
```

### 4. Documentos

Los documentos ya están incluidos en `data/docs/`. Para descargarlos de nuevo desde el repositorio oficial de FastAPI:

```bash
mkdir -p data/docs
BASE=https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial
for f in handling-errors path-params query-params body cors background-tasks middleware; do
  curl -sL "$BASE/$f.md" -o "data/docs/$f.md"
done
curl -sL "$BASE/dependencies/index.md" -o data/docs/dependencies.md
```

### 5. Ejecución

```bash
python setup_index.py   # crea el índice Serverless (384 dim, cosine, aws us-east-1)
python ingest.py        # chunking + embeddings + subida a Pinecone
python rag_system.py    # consulta de prueba
python evaluate.py      # métricas
```

La primera ejecución descarga el modelo de embeddings (~90 MB).

## Resultados de la evaluación

Golden set de 5 preguntas, top-5 recuperado:

| Modo | Recall@5 | Precision@5 |
|---|---|---|
| BM25 | 100% | 60% |
| Vectorial | 100% | 56% |
| **Híbrido** | **100%** | **60%** |

### Detalle del híbrido

| Pregunta | Documento esperado | Chunks del documento | Máximo posible | Precision@5 |
|---|---|---|---|---|
| allow_origins | `cors.md` | 3 | 60% | 60% |
| BackgroundTasks | `background-tasks.md` | 3 | 60% | 60% |
| X-Process-Time | `middleware.md` | 2 | 40% | 40% |
| Optional query parameter | `query-params.md` | 3 | 60% | 60% |
| Pydantic BaseModel | `body.md` | 4 | 80% | 80% |

### Análisis

- **Recall@5 de 100%:** en todas las preguntas, el documento correcto aparece entre los 5 resultados.
- **Precision@5 de 60% es el máximo alcanzable con este dataset.** Como cada pregunta tiene un solo documento relevante, la precisión está limitada por la cantidad de chunks de ese documento (por ejemplo, `middleware.md` tiene 2 chunks, así que su máximo es 2/5 = 40%). El recuperador híbrido trajo **todos** los chunks del documento correcto en cada pregunta.
- **Híbrido vs. vectorial:** el híbrido supera al vectorial (60% vs. 56%) porque las preguntas incluyen términos técnicos exactos (`allow_origins`, `BackgroundTasks`, `X-Process-Time`), donde la búsqueda léxica de BM25 es más precisa que la semántica.
- **Limitación:** el golden set es chico (5 preguntas) y con un único documento relevante por pregunta, por lo que Recall@5 solo puede valer 0 o 1 en cada caso. Un benchmark más grande y con varios documentos relevantes por pregunta permitiría diferenciar mejor los tres modos.