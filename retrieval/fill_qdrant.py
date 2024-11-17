from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import pickle

# клиент (поднят в докере)
client = QdrantClient(url="http://localhost:6333")

# загрузка эмбедов и соответствующих тектовых чанков
chunks_texts = pickle.load(open('cap_daughter_chunks.pickle', 'rb'))
chunks_vectors = pickle.load(open('chunks_vectors.pickle', 'rb'))
chunks_vectors = chunks_vectors.toarray()

# создание хранилища
collection_name = "cap_daughter"
client.create_collection(
    collection_name=collection_name,
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

client.upsert(collection_name=collection_name, points=points)
print('successfully loaded!')