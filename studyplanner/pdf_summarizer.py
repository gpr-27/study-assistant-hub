import os
import json
from PyPDF2 import PdfReader
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class PDFSummarizer:
    def __init__(self):
        """Initialize the PDF summarizer."""
        self.summaries = {}  # Store summaries in memory
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def generate_summary(self, pdf_text, pdf_name):
        """Generate a summary of the PDF content using OpenAI API."""
        try:
            # Create a prompt for the LLM
            prompt = f"""
            Analyze the following lecture content and provide:
            1. A list of main topics (3-5 bullet points)
            2. Key concepts for each topic
            3. Estimated difficulty level (Basic/Intermediate/Advanced) for each topic
            4. A brief summary of the overall content
            
            Format the response as follows:
            TOPICS:
            • [Topic 1] - [Difficulty]
              - [Key concept 1]
              - [Key concept 2]
            • [Topic 2] - [Difficulty]
              - [Key concept 1]
              - [Key concept 2]
            ...
            
            SUMMARY:
            [Overall summary of the content]
            
            PDF: {pdf_name}
            
            CONTENT:
            {pdf_text[:50000]}  # Limiting content to avoid token limits
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Can be changed to other models as needed
                messages=[
                    {"role": "system", "content": "You are an educational assistant that summarizes lecture content effectively."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Failed to generate summary."
    
    def save_summary(self, subject, pdf_name, summary):
        """Save the PDF summary to memory."""
        if subject not in self.summaries:
            self.summaries[subject] = {}
        self.summaries[subject][pdf_name] = summary
        return summary
    
    def summarize_pdf(self, pdf_path, subject):
        """Process a PDF file: extract text and generate summary."""
        pdf_name = os.path.basename(pdf_path)
        print(f"Processing {pdf_name}...")
        
        # Check if summary exists in memory
        if subject in self.summaries and pdf_name in self.summaries[subject]:
            print(f"Summary for {pdf_name} already exists in memory. Using cached version...")
            return self.summaries[subject][pdf_name]
        
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return None
        
        # Generate summary
        summary = self.generate_summary(pdf_text, pdf_name)
        
        # Save summary in memory
        self.save_summary(subject, pdf_name, summary)
        
        return summary
    
    def summarize_pdfs(self, pdf_paths, subject):
        """Summarize multiple PDFs and return a dictionary of their summaries."""
        results = {}
        for pdf_path in pdf_paths:
            summary = self.summarize_pdf(pdf_path, subject)
            if summary:
                pdf_name = os.path.basename(pdf_path)
                results[pdf_name] = summary
        
        return results
    
    def get_saved_summaries(self, subject):
        """Get all saved summaries for a subject from memory."""
        return self.summaries.get(subject, {})


if __name__ == "__main__":
    # Example usage
    summarizer = PDFSummarizer()
    
    # Example subject and PDFs
    subject = "dccn"
    documents_dir = os.path.join("documents", subject)
    pdf_files = [os.path.join(documents_dir, f) for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    
    # Summarize PDFs
    summaries = summarizer.summarize_pdfs(pdf_files, subject)
    print(f"Successfully summarized {len(summaries)} PDFs for {subject}")
