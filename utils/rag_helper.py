# from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from transformers import pipeline
# from langchain_community.llms import HuggingFacePipeline

# INDEX_PATH = "faiss_index"

# def get_llm():
#     pipe = pipeline(
#     "text-generation",
#     model="google/flan-t5-base",
#     max_new_tokens=256
# )
#     llm = HuggingFacePipeline(pipeline=pipe)
#     return llm


# def load_vector_db():
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     vector_db = FAISS.load_local(
#         INDEX_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     return vector_db


# def retrieve_context(query, k=3):
#     vector_db = load_vector_db()
#     docs = vector_db.similarity_search(query, k=k)
#     return docs


# def rag_answer(query):
#     docs = retrieve_context(query)

#     context = "\n\n".join([doc.page_content for doc in docs])

#     prompt = f"""
# Context:
# {context}

# Question: {query}

# Give a short direct answer based only on the context.
# If not found, say: I don't know.
# """

#     llm = get_llm()
#     response = llm.invoke(prompt)
#     return response

# utils/rag_helper.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

INDEX_PATH = "faiss_index"


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_db


def get_llm():
    return OllamaLLM(model="mistral")


def rag_answer(query, k=4):
    vector_db = load_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": k})

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an HR Policy Assistant.
Answer ONLY using the given context.
Give exact answer with numbers/timings.
If answer not found, say: I don't know.

Context:
{context}

Question: {query}

Answer:
"""

    llm = get_llm()
    return llm.invoke(prompt).strip()