from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import io
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

chunks = []


# -------- PDF TEXT EXTRACTION -------- #
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


# -------- TEXT CHUNKING -------- #
def chunk_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


# -------- FINAL SMART ANSWER FUNCTION -------- #
def get_answer(question):
    global chunks

    # 🔥 stopwords
    stopwords = {
        "what","is","the","who","are","our","of",
        "in","on","for","a","an","to","and"
    }

    keywords = [
        word.lower() for word in question.split()
        if word.lower() not in stopwords
    ]

    if not keywords:
        return "Answer not found in document"

    # Step 1: score chunks based on keyword match
    scored_chunks = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in keywords if word in chunk_lower)

        if score > 0:
            scored_chunks.append((chunk, score))

    # Step 2: no match
    if not scored_chunks:
        return "Answer not found in document"

    # Step 3: take best matching chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [c[0] for c in scored_chunks[:5]]

    # Step 4: semantic search on top chunks
    embeddings = model.encode(top_chunks, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    q_embedding = model.encode([question], convert_to_numpy=True)
    q_embedding = q_embedding / np.linalg.norm(q_embedding)

    temp_index = faiss.IndexFlatIP(embeddings.shape[1])
    temp_index.add(embeddings)

    scores, indices = temp_index.search(q_embedding, 1)

    answer = top_chunks[indices[0][0]]

    return answer


# -------- ROUTES -------- #

@app.route("/")
def home():
    return "RAG Backend Running 🚀"


@app.route("/upload", methods=["POST"])
def upload():
    global chunks

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF allowed"}), 400

    pdf_bytes = io.BytesIO(file.read())

    text = extract_text(pdf_bytes)

    if not text:
        return jsonify({"error": "Could not extract text"}), 400

    chunks = chunk_text(text)

    if not chunks:
        return jsonify({"error": "No usable content"}), 400

    return jsonify({
        "message": "PDF uploaded successfully",
        "total_chunks": len(chunks)
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    if not chunks:
        return jsonify({"error": "Upload PDF first"}), 400

    answer = get_answer(data["question"])

    return jsonify({
        "question": data["question"],
        "answer": answer
    })


@app.route("/health")
def health():
    return jsonify({"status": "running"})


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)