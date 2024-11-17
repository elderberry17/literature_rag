from qdrant_client import QdrantClient
import pickle
from retrieval.qdrant_conf import QDRANT_URL, COLLECTION_NAME, VECTORIZER_PATH

def vectorize_text(text, vectorizer):
    embed = vectorizer.transform([text]).toarray().reshape(-1).tolist()
    return embed

def find_chunks(text, vectorizer, collection_name, client, limit=1):
    embed = vectorize_text(text, vectorizer)
    search_results = client.query_points(
        collection_name=collection_name,
        query=embed,
        limit=limit
    ).points

    chunks = [result.payload['source_text'] for result in search_results]
    return chunks

