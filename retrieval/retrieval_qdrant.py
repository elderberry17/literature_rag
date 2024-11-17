from qdrant_client import QdrantClient
import pickle

# load client
client = QdrantClient(url="http://localhost:6333")
collection_name = "cap_daughter"

# load vectorizer
tfidf_vec = pickle.load(open('tfidf_vec.pickle', 'rb'))

def vectorize_text(text, vectorizer):
    embed = vectorizer.transform([text]).toarray().reshape(-1).tolist()
    return embed

def find_chunks(text, vectorizer, collection_name, limit):
    embed = vectorize_text(text, vectorizer)
    search_results = client.query_points(
        collection_name=collection_name,
        query=embed,
        limit=limit
    ).points

    chunks = [result.payload['source_text'] for result in search_results]
    return chunks


if __name__ == "__main__":
    query = 'какую сказу рассказал пугачев гриневу?'
    top_chunk = find_chunks(query, tfidf_vec, collection_name, limit=1)
    print(top_chunk)
