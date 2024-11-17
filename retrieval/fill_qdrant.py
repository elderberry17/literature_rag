from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import pickle
from qdrant_conf import QDRANT_URL, CHUNKS_PATH, VECTORS_PATH, COLLECTION_NAME


if __name__ == "__main__":
    # клиент (поднят в докере)
    client = QdrantClient(url=QDRANT_URL)

    # загрузка эмбедов и соответствующих тектовых чанков
    chunks_texts = pickle.load(open(CHUNKS_PATH, 'rb'))
    chunks_vectors = pickle.load(open(VECTORS_PATH, 'rb'))
    chunks_vectors = chunks_vectors.toarray()

    # создание хранилища
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    ## код для заполнения хранилища 
    points = [
        {
            "id": i,
            "vector": embedding,
            "payload": {"source_text": text}  # Add source text as metadata
        }
        for i, (embedding, text) in enumerate(zip(chunks_vectors, chunks_texts))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print('successfully loaded!')