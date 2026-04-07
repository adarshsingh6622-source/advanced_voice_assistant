from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

dimension = 384
index = faiss.IndexFlatL2(dimension)

memory_text = []

def store_memory(text):

    embedding = model.encode([text])

    index.add(np.array(embedding))

    memory_text.append(text)

def search_memory(query):

    query_vector = model.encode([query])

    D, I = index.search(np.array(query_vector), k=1)

    if len(memory_text) > 0:
        return memory_text[I[0][0]]

    return ""