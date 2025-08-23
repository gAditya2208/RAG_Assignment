from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def simple_sentence_split(text, max_chunk_size=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Add the period back because split removes it
        sentence = sentence if sentence.endswith('.') else sentence + '.'

        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def create_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

if __name__ == "__main__":
    from ingest_pdf import extract_text_from_pdf

    pdf_file = r"C:\Users\gadit\Desktop\RAG_Assignment\data\What_is_Policy.pdf"
    texts = extract_text_from_pdf(pdf_file)
    all_text = " ".join(texts).replace("\n", " ")

    chunks = simple_sentence_split(all_text)

    index, embeddings = create_faiss_index(chunks)
    print(f"Number of chunks indexed: {len(chunks)}")
