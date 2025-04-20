import os
import json
import datetime
from dateutil import parser
import openai
from dotenv import load_dotenv

from pdf_summarizer import PDFSummarizer

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class StudyPlanGenerator:
    def __init__(self):
        """Initialize the study plan generator."""
        self.summarizer = PDFSummarizer()
    
    def get_pdf_paths(self, subject):
        """Get paths of all PDF files for a given subject."""
        documents_dir = os.path.join("documents", subject)
        if not os.path.exists(documents_dir):
            return []
        
        return [os.path.join(documents_dir, f) for f in os.listdir(documents_dir) 
                if f.endswith('.pdf')]
    
    def get_pdf_list(self, subject):
        """Get a list of available PDF filenames for a subject."""
        documents_dir = os.path.join("documents", subject)
        if not os.path.exists(documents_dir):
            return []
        
        return [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    
    def ensure_summaries(self, subject, pdf_names):
        """Ensure summaries exist for the selected PDFs."""
        documents_dir = os.path.join("documents", subject)
        pdf_paths = [os.path.join(documents_dir, pdf_name) for pdf_name in pdf_names]
        
        # Generate summaries for PDFs that don't have them
        self.summarizer.summarize_pdfs(pdf_paths, subject)
        
        # Return the summaries for selected PDFs
        all_summaries = self.summarizer.get_saved_summaries(subject)
        selected_summaries = {pdf: all_summaries.get(pdf, "") for pdf in pdf_names}
        
        return selected_summaries
    
    def generate_study_plan(self, subject, pdf_names, target_date, days_to_study=None):
        """Generate a study plan for the selected PDFs with a target completion date."""
        # Ensure we have summaries for all selected PDFs
        summaries = self.ensure_summaries(subject, pdf_names)
        
        # Parse the target date
        try:
            target_date = parser.parse(target_date).date()
            start_date = datetime.date.today()
            
            # Calculate days available for study
            if days_to_study is None:
                delta = target_date - start_date
                days_to_study = max(1, delta.days)  # Ensure at least 1 day
        except Exception as e:
            print(f"Error parsing date: {e}")
            return None
        
        # Prepare content for the LLM
        content_for_plan = "\n\n".join([
            f"PDF: {pdf_name}\nSummary: {summary}" 
            for pdf_name, summary in summaries.items()
        ])
        
        try:
            # Create a prompt for the LLM
            prompt = f"""
            Create a detailed day-by-day study plan for {days_to_study} days, starting from {start_date} to {target_date}.
            The plan should cover the following lecture materials for the subject "{subject}":

            PDF Summaries:
            {content_for_plan}

            Guidelines:
            1. Use the topic information and difficulty levels from the summaries to organize the study plan
            2. Start with basic topics and progress to more advanced ones
            3. Include specific study techniques based on topic difficulty
            4. Add short breaks and review sessions
            5. IMPORTANT: Each day MUST include AT LEAST 2 HOURS of study time

            FORMAT THE PLAN AS FOLLOWS:
            # Study Plan for {subject}

            Duration: {days_to_study} days | Exam Date: {target_date}

            Key Objectives:
            - [List 3-5 key objectives based on the topics]
            - ...

            ## Day 1: {start_date.isoformat()} (2 hours)
            - [Topic to study with PDF reference]
            - [Study technique]
            - [Break/Review timing]

            ## Day 2: [date] (2 hours)
            [Continue for all days]

            Make the plan progressive and include regular reviews.
            Remember that EACH DAY MUST HAVE AT LEAST 2 HOURS of dedicated study time.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # Can be changed to other models as needed
                messages=[
                    {"role": "system", "content": "You are an educational expert who creates effective study plans with at least 2 hours of study time per day."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000
            )
            
            study_plan = response.choices[0].message.content
            return study_plan
        
        except Exception as e:
            print(f"Error generating study plan: {e}")
            return "Failed to generate study plan."
