from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/policy.pdf"
INDEX_PATH = "faiss_index"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80
)
chunks = splitter.split_documents(docs)

# 3. HuggingFace Embedding Model (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store embeddings in FAISS
vector_db = FAISS.from_documents(chunks, embeddings)

# 5. Save FAISS index
vector_db.save_local(INDEX_PATH)

print("✅ FAISS index created successfully using HuggingFace embeddings!")