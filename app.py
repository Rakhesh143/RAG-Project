from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app)

# ✅ LIGHTWEIGHT MODEL (important for Render free tier)
print("Loading model...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print("Model loaded!")

chunks = []
index = None

# -------- PDF PROCESS -------- #
def process_pdf(file):
    global chunks, index

    pdf_reader = PyPDF2.PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # ✅ smaller chunks (less memory)
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))


# -------- ROUTES -------- #

@app.route("/")
def home():
    return "RAG PDF QA is running!"

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    process_pdf(file)
    return jsonify({"message": "PDF processed", "chunks": len(chunks)})

@app.route("/ask", methods=["POST"])
def ask():
    global index, chunks

    data = request.json
    query = data.get("query")

    if index is None:
        return jsonify({"answer": "Upload PDF first"})

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)

    results = [chunks[i] for i in I[0]]
    answer = " ".join(results)

    if not answer.strip():
        answer = "Answer not found in document"

    return jsonify({"answer": answer})


# -------- RUN -------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)