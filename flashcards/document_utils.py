from pathlib import Path
from PyPDF2 import PdfReader

def get_subjects():
    """Get list of subjects from the documents folder"""
    docs_path = Path("documents")
    
    # Skip if documents folder doesn't exist
    if not docs_path.exists():
        return []
    
    # Get subject folders (skip hidden folders)
    subjects = []
    for folder in docs_path.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            subjects.append(folder.name)
            
    return subjects

def get_pdfs_for_subject(subject):
    """Get PDF files available for a subject"""
    subject_path = Path("documents") / subject
    
    # Skip if subject folder doesn't exist
    if not subject_path.exists():
        return []
    
    # Get PDF files
    pdfs = []
    for file in subject_path.iterdir():
        if file.is_file() and file.suffix.lower() == '.pdf':
            pdfs.append(file.name)
            
    return pdfs

def read_pdf(subject, filename):
    """Read and get text from a PDF file"""
    pdf_path = Path("documents") / subject / filename
    
    # Make sure file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"Can't find PDF file: {filename}")
    
    # Read PDF content
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        # Get text from each page
        for page in reader.pages:
            text += page.extract_text()
            
        return text.strip()
        
    except Exception as e:
        raise Exception(f"Had trouble reading the PDF: {str(e)}")

def get_pdf_pages(subject, filename):
    """Get total pages in the PDF"""
    pdf_path = Path("documents") / subject / filename
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Can't find PDF file: {filename}")
    
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        raise Exception(f"Had trouble reading the PDF: {str(e)}")
