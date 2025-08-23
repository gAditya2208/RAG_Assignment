from flask import Flask, request, render_template
from retriever import retrieve_answer
from build_index import simple_sentence_split, create_faiss_index
from ingest_pdf import extract_text_from_pdf

app = Flask(__name__)

pdf_file = "data/What_is_Policy.pdf"
text_pages = extract_text_from_pdf(pdf_file)  # Extract text from PDF
all_text = " ".join(text_pages).replace("\n", " ")  # Combine and clean text
chunks = simple_sentence_split(all_text)  # Chunk based on sentences
index, embeddings = create_faiss_index(chunks)

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = ""
    if request.method == 'POST':
        question = request.form['question']
        results = retrieve_answer(question, index, chunks, embeddings)
        answer = "\n\n".join(results)
    return render_template('index.html', answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
