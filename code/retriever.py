import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_answer(question, index, chunks, embeddings, top_k=3):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding).astype('float32'), top_k)
    results = [chunks[i] for i in I[0]]
    return results

if __name__ == "__main__":
    from ingest_pdf import extract_text_from_pdf
    from build_index import simple_sentence_split, create_faiss_index

    pdf_file = "data/What_is_Policy.pdf"
    text_pages = extract_text_from_pdf(pdf_file)
    all_text = " ".join(text_pages).replace("\n"," ")
    chunks = simple_sentence_split(all_text)

    index, embeddings = create_faiss_index(chunks)

    question = "What is policy?"
    answers = retrieve_answer(question, index, chunks, embeddings)
    for idx, ans in enumerate(answers):
        print(f"Answer chunk {idx+1}:\n{ans}\n")
