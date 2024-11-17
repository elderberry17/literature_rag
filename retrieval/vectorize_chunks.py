import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from qdrant_conf import CHUNKS_PATH, VECTORS_PATH, VECTORIZER_PATH

if __name__ == "__main__":
    ## это тектовые чанки (норм вроде нарезаны)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)

    # бейзлайн с простой векторизацей с tf-idf (можно с bm25 или типа того)
    tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=768)
    tfidf_vec.fit(chunks)

    # сохраняем векторизатор для инференса
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vec, f)

    # делаем эмбеддинги и их тоже сохраняем
    chunks_vectors = tfidf_vec.transform(chunks)
    with open(VECTORS_PATH, 'wb') as f:
        pickle.dump(chunks_vectors, f)