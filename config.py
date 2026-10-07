import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # opcional: no se usa con embeddings locales
INDEX_NAME = os.getenv("INDEX_NAME", "fastapi-docs-rag")
NAMESPACE = "fastapi-docs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

DATA_DIR = "data/docs"
CHUNKS_PATH = "chunks.json"
TOP_K = 5