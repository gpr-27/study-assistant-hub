import streamlit as st
from pathlib import Path
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from utils.document_loader import load_single_pdf
from utils.qa_chain import create_qa_chain

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "db" not in st.session_state:
    st.session_state.db = None

if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

# Add CSS to fix white line and ensure proper spacing
st.markdown("""
<style>
.stChatMessage {
    margin-bottom: 0px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Study Buddy Chatbot 📚")

# Sidebar for document selection
st.sidebar.title("Document Selection")
doc_option = st.sidebar.radio(
    "Choose your study material:",
    ["Select Subject", "Upload PDF"]
)

def load_faiss_index():
    if (Path(__file__).parent.parent / "faiss_index").exists():
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return FAISS.load_local(str(Path(__file__).parent.parent / "faiss_index"), embeddings, allow_dangerous_deserialization=True)
    return None

def handle_subject_selection():
    subjects = []
    documents_path = Path(__file__).parent.parent / "documents"
    if documents_path.exists():
        subjects = [f.name for f in documents_path.iterdir() if f.is_dir()]
    
    if not subjects:
        st.sidebar.warning("No subjects found.")
        return
    
    selected_subject = st.sidebar.selectbox("Select Subject:", subjects)
    
    if st.sidebar.button("Load Subject"):
        with st.spinner(f"Loading {selected_subject} materials..."):

            # Load the index
            db = load_faiss_index()
            if db:
                # Filter for the selected subject
                retriever = db.as_retriever(
                    search_kwargs={
                        "filter": {"subject": selected_subject}
                    }
                )
                st.session_state.qa_chain, st.session_state.memory = create_qa_chain(retriever)
                st.sidebar.success(f"Loaded {selected_subject} materials!")
            else:
                st.sidebar.error("Failed to load the document index.")

def handle_pdf_upload():
    uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file:
        # Check if it's a new file
        is_new_file = st.session_state.current_pdf_name != uploaded_file.name
        
        # Process only if it's a new file or no database exists
        if is_new_file or st.session_state.db is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("Processing your PDF..."):
                # Create new database and QA chain
                st.session_state.db = load_single_pdf(tmp_path)
                retriever = st.session_state.db.as_retriever()
                st.session_state.qa_chain, st.session_state.memory = create_qa_chain(retriever)
                st.session_state.current_pdf_name = uploaded_file.name
                # Reset message history for new file
                if is_new_file:
                    st.session_state.messages = []
                
            Path(tmp_path).unlink()  # Clean up the temporary file
            st.sidebar.success("PDF processed successfully!")
        # If same file is uploaded again, keep existing QA chain and memory

# Handle document selection
if doc_option == "Select Subject":
    handle_subject_selection()
else:
    handle_pdf_upload()

# Chat interface
if st.session_state.get("qa_chain"):
    # Handle new input
    if prompt := st.chat_input("Ask a question about your study materials"):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display all messages including the new user message
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Show thinking message while generating response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Generate response
                    response = st.session_state.qa_chain({"question": prompt})
                    
                    # Add assistant response to state
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response["answer"]}
                    )
                    
                    # Force a rerun to update the display with the new response
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred while generating a response: {e}")
    else:
        # Display existing chat history when no new input
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

else:
    st.info("Please select a subject or upload a PDF to start chatting!")
