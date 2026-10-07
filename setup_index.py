import time
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME, EMBEDDING_DIM


def setup_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existentes = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME in existentes:
        print(f"♻️ El índice '{INDEX_NAME}' ya existe")
    else:
        print(f"🆕 Creando índice '{INDEX_NAME}' (dimensión {EMBEDDING_DIM})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Esperar a que esté listo antes de usarlo
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("✅ Índice listo")

    print(pc.Index(INDEX_NAME).describe_index_stats())


if __name__ == "__main__":
    setup_index()