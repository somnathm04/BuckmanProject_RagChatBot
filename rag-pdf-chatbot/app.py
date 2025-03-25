import streamlit as st
import fitz  # PyMuPDF for PDFs
import pandas as pd
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Local HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read()) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# Function to extract text from a CSV file
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)  # Convert DataFrame to a text format

# Function to extract text from a DOCX file
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to process uploaded files
def extract_text_from_files(uploaded_files):
    extracted_texts = []
    
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            extracted_texts.append(extract_text_from_pdf(file))
        elif file.name.endswith(".csv"):
            extracted_texts.append(extract_text_from_csv(file))
        elif file.name.endswith(".docx"):
            extracted_texts.append(extract_text_from_docx(file))
        else:
            st.error(f"Unsupported file type: {file.name}")
    
    return extracted_texts

# Function to split and store text embeddings in FAISS
def get_vector_store(text_chunks):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_texts = text_splitter.split_text("\n\n".join(text_chunks))

    # Store embeddings in FAISS
    vector_store = FAISS.from_texts(split_texts, embedding=embedding_model)  
    return vector_store

# Function to process user input
def user_input(user_question, uploaded_files, conversation_history):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return
    
    # Extract text from files
    text_chunks = extract_text_from_files(uploaded_files)
    
    # Get FAISS Vector Store
    vector_store = get_vector_store(text_chunks)

    # Retrieve the most relevant document chunk
    docs = vector_store.similarity_search(user_question, k=3)
    response = "\n\n".join([doc.page_content for doc in docs])

    # Store conversation history
    conversation_history.append({"question": user_question, "response": response})
    
    # Display response
    st.write(response)

# Streamlit UI
st.title("RAG Chatbot (PDF, CSV, DOCX) - Local Model")
user_question = st.text_input("Ask a question about the document(s):")
uploaded_files = st.file_uploader("Upload files (PDF, CSV, DOCX)", type=["pdf", "csv", "docx"], accept_multiple_files=True)

# Store conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if st.button("Submit"):
    user_input(user_question, uploaded_files, st.session_state.conversation_history)

# Show conversation history
if st.session_state.conversation_history:
    st.subheader("Conversation History")
    for chat in st.session_state.conversation_history:
        st.write(f"*Q:* {chat['question']}")
        st.write(f"*A:* {chat['response']}")
        