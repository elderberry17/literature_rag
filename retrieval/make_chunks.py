import re
from qdrant_conf import BOOK_PATH, CHUNKS_PATH
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
nltk.download('punkt_tab')
import pickle

# для нарезки с перекрытием
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # грузим текст
    with open(BOOK_PATH, 'r') as f:
        data = f.read()

    # урезаем аннотацию
    data = data[611:]

    # паттерн для сплита и сплит по главам
    chapter_pattern = r"(Глава\s+[IVXLCDM]+)\n\n([^\n]+)"
    chapters = re.split(chapter_pattern, data)

    # структурируем чанки
    structured_chapters = []
    for i in range(1, len(chapters), 3):
        title = chapters[i].strip()         # Chapter title (e.g., "Глава I")
        name = chapters[i + 1].strip()       # Chapter name (e.g., "Сержант гвардии[1]")
        content = chapters[i + 2].strip()    # Chapter content
        structured_chapters.append({
            "title": title,
            "name": name,
            "content": content
        })

    # теперь внутри каждой главы режем чанки (чтобы не пересекать главы)
    CHUNKS_SIZE = 1000
    OVERLAP_SIZE = 400
    chunks_langchain = []
    for chapter in structured_chapters:
        chapter_content = chapter['content'].replace('\n', ' ')
        chapter_chunks = chunk_text(chapter_content, CHUNKS_SIZE, OVERLAP_SIZE)
        chapter_chunks = [f'{chapter["title"]}: {chapter["name"]}\n{chunk}' for chunk in chapter_chunks]
        chunks_langchain.extend(chapter_chunks)

    # сохраняем
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks_langchain, f)
    print('chunks saved successfully!')




