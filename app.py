# import streamlit as st
# from utils.rag_helper import rag_answer

# st.set_page_config(page_title="Free RAG Chatbot", layout="wide")

# st.title("📄 Free PDF Chatbot (HuggingFace + FAISS)")

# query = st.text_input("Ask a question:")

# if query:
#     answer = rag_answer(query)
#     st.write("### Answer:")
#     st.write(answer)

import streamlit as st
from utils.rag_helper import rag_answer

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📄 PDF RAG Chatbot (FAISS + Ollama Mistral)")

st.sidebar.title("⚙️ Settings")
st.sidebar.write("This chatbot answers questions from your PDF using FAISS + Ollama Mistral.")
st.sidebar.markdown("### 🔍 Model Used")
st.sidebar.write("Embedding: all-MiniLM-L6-v2")
st.sidebar.write("LLM: mistral (Ollama)")

# session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# chat input
query = st.chat_input("Type your question here...")

if query:
    # user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # bot response
    answer = rag_answer(query)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)