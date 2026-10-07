import json
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNKS_PATH, DATA_DIR, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE,
)

# Etiqueta de categoría por archivo (metadata avanzada)
CATEGORIAS = {
    "path-params.md": "parametros",
    "query-params.md": "parametros",
    "body.md": "request",
    "handling-errors.md": "errores",
    "cors.md": "seguridad",
    "middleware.md": "middleware",
    "background-tasks.md": "tareas",
    "dependencies.md": "dependencias",
}


def cargar_y_dividir():
    loader = DirectoryLoader(
        DATA_DIR, glob="*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,     # dentro del rango 500-800 tokens
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    # Numerar los chunks dentro de cada archivo
    contador = {}
    for chunk in chunks:
        fuente = os.path.basename(chunk.metadata["source"])
        idx = contador.get(fuente, 0)
        contador[fuente] = idx + 1

        chunk.metadata = {
            "source": fuente,
            "category": CATEGORIAS.get(fuente, "general"),
            "chunk_index": idx,
        }
    return chunks


def guardar_chunks(chunks):
    """BM25 corre en memoria: guardamos los chunks para reconstruirlo después."""
    data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def subir_a_pinecone(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, embedding=embeddings, namespace=NAMESPACE,
    )
    # IDs fijos: si corrés la ingesta de nuevo, se sobrescriben (no se duplican)
    ids = [f"{c.metadata['source']}-{c.metadata['chunk_index']}" for c in chunks]
    vectorstore.add_documents(chunks, ids=ids)


if __name__ == "__main__":
    chunks = cargar_y_dividir()
    print(f"✂️ {len(chunks)} chunks generados")
    print("Metadata de ejemplo:", chunks[0].metadata)

    guardar_chunks(chunks)
    print(f"💾 Chunks guardados en {CHUNKS_PATH}")

    subir_a_pinecone(chunks)
    print(f"📦 Subidos a Pinecone (namespace '{NAMESPACE}')")