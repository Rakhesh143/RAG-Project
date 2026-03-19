# 📄 AI-Powered PDF Question Answering System (RAG)

## 🚀 Project Overview

This project is an **AI-powered PDF Question Answering System** built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload a PDF document and ask questions, and the system retrieves relevant content and generates accurate answers.

The goal of this project is to demonstrate real-world application of **Generative AI, NLP, and semantic search**.

---

## 🧠 Key Features

* 📂 Upload and process PDF documents
* 🔍 Intelligent text chunking and preprocessing
* 🤖 Semantic search using embeddings
* 📊 Fast similarity search using FAISS
* 💬 Question answering using LLM APIs
* ⚡ Real-time response generation
* 🌐 Flask-based backend API

---

## 🛠️ Tech Stack

* **Python**
* **Flask**
* **FAISS (Vector Database)**
* **Sentence Transformers**
* **OpenAI API / LLM APIs**
* **PyPDF2**
* **NumPy**

---

## 🔄 How It Works

1. Upload PDF
2. Extract text from document
3. Split into smaller chunks
4. Convert chunks into embeddings
5. Store embeddings in FAISS index
6. User asks a question
7. Relevant chunks are retrieved
8. LLM generates final answer

---

## ▶️ Run Locally (Website Mode)

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🌍 Deployment Attempt & Challenges Faced

This project was also prepared for **cloud deployment (Render)**, and during this process several real-world challenges were encountered:

### ⚠️ Challenges

* ❌ **Port binding issue**
  Initially faced error:

  ```
  No open ports detected
  ```

  Fixed by correctly binding Flask app to `0.0.0.0` and using environment port.

* ⚠️ **Heavy ML model loading issue**
  The SentenceTransformer model (`all-MiniLM-L6-v2`) is relatively heavy, which caused:

  * Slow startup time
  * Deployment timeouts

* ⚠️ **Memory constraints**
  Free-tier cloud services have limited RAM, which affected:

  * Model loading
  * FAISS indexing

* ⚠️ **Cold start delays**
  Application took time to respond initially due to model initialization.

---

### ✅ Solutions Tried

* Used **lighter embedding model**
* Optimized chunk size and processing
* Adjusted Flask app configuration
* Attempted **Gunicorn for production server**
* Ensured correct environment variable usage (`PORT`)

---

## 💡 Key Learnings

* Practical understanding of **RAG architecture**
* Hands-on experience with **LLM APIs and embeddings**
* Learned deployment concepts (ports, servers, scaling)
* Debugging real-world production issues
* Handling performance constraints in AI systems

---

## 📌 Future Improvements

* Add frontend UI (React / Streamlit)
* Optimize model loading using caching
* Deploy using scalable cloud (AWS / Azure)
* Add authentication system
* Improve answer accuracy with better prompting

---

## 👨‍💻 Author

**Rakesh Namineni**
📧 Email: [naminenirakesh@gmail.com](mailto:naminenirakesh@gmail.com)
🔗 LinkedIn: https://www.linkedin.com/in/rakesh-namineni-688062291
💻 GitHub: https://github.com/Rakhesh143

---

## ⭐ Conclusion

This project demonstrates my ability to:

* Build end-to-end AI applications
* Work with LLMs and RAG systems
* Solve real-world deployment challenges


## 🎥 Demo
This application currently runs locally due to deployment limitations with heavy ML models.

To test:
- Run: `python app.py`
- Open: http://127.0.0.1:5000/

> Note: Deployment was attempted on Render, but faced challenges with model size, memory limits, and startup delays.

---

