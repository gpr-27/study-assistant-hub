from pathlib import Path
from .document_loader import load_folder_documents

def build_and_save_index():
    try:
        # Get paths relative to project root
        root_path = Path(__file__).parent.parent.parent
        documents_path = root_path / "documents"
        faiss_path = root_path / "faiss_index"
        
        # Load all documents and create FAISS index
        print(f"Loading documents from {documents_path}...")
        db = load_folder_documents(documents_path)
        
        # Save the index
        print("Saving FAISS index...")
        db.save_local(str(faiss_path))
        
    except Exception as e:
        print(f"Error building index: {str(e)}")

if __name__ == "__main__":
    build_and_save_index()
