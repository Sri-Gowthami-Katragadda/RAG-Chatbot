# 📄 PDF RAG Chatbot (FAISS + HuggingFace Embeddings + Ollama)

This project is a **PDF-based RAG (Retrieval Augmented Generation) Chatbot** built using **LangChain, FAISS, HuggingFace Embeddings, Ollama (Mistral), and Streamlit**.

It allows users to ask questions from a PDF document (example: HR policy PDF). The chatbot retrieves relevant PDF content using **vector similarity search** and generates accurate answers using a **local LLM**.

---

## 🚀 Features

- 📄 Load and process PDF documents
- ✂️ Split PDF into meaningful chunks
- 🔍 Generate embeddings using HuggingFace embedding model
- 🧠 Store embeddings in FAISS vector database
- 🤖 Answer questions using Ollama local LLM (Mistral)
- 💬 Streamlit chatbot UI with chat history
- 🆓 Fully Free (No OpenAI / Paid API)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- Ollama
- Mistral LLM

---

## 📂 Project Structure
rag_chatbot/
│
├── app.py
├── ingest.py
├── requirements.txt
│
├── data/
│ └── policy.pdf
│
├── faiss_index/
│ ├── index.faiss
│ └── index.pkl
│
└── utils/
├── init.py
└── rag_helper.py

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
2️⃣ Install Dependencies
pip install -r requirements.txt
🧠 Install Ollama (Local LLM)
1️⃣ Install Ollama

Download from:
👉 https://ollama.com/download

Check installation:

ollama --version
2️⃣ Download Mistral Model
ollama pull mistral

Test:

ollama run mistral
📌 Step 1: Add PDF File

Place your PDF inside:

data/policy.pdf
📌 Step 2: Create FAISS Index (Ingest PDF)

Run:

python ingest.py

This will generate:

faiss_index/
📌 Step 3: Run Streamlit App
streamlit run app.py

The chatbot will run on:

http://localhost:8501

🧪 Example Questions
What is the leave policy?
What are the working hours?
What are the working days?
What is the travel policy?
What is the dress code policy?
🧠 How RAG Works (Simple Explanation)
PDF is loaded and split into chunks
Each chunk is converted into embeddings
Embeddings are stored inside FAISS vector database
When user asks a question, FAISS retrieves relevant chunks
Retrieved chunks are passed as context to Mistral LLM
LLM generates accurate answer from context
⚡ Notes
First response may be slow because Ollama loads the model into memory.
After that, responses become faster.
This project runs completely offline after model download.