import os
import sys
import datetime
import streamlit as st
import re

# Add parent directory to sys.path to import from the same directory level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studyplanner.study_plan_generator import StudyPlanGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="AI Study Buddy - Study Planner",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
.plan-header {
    background-color: #1E3A8A;
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.objectives {
    background-color: #2563EB;
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
}
.day-card {
    background-color: #F3F4F6;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 4px solid #2563EB;
}
.day-header {
    color: #1E3A8A;
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 10px;
}
.activity {
    padding: 5px 0;
    color: #374151;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'generator' not in st.session_state:
    st.session_state.generator = StudyPlanGenerator()

def parse_study_plan(plan_text):
    """Parse the study plan text into structured sections."""
    # Extract title and metadata
    title_match = re.search(r'#\s+(.*?)(?=\n|$)', plan_text)
    title = title_match.group(1) if title_match else "Study Plan"
    
    # Extract duration and date
    metadata = re.search(r'Duration:\s*(\d+)\s*days.*?Exam Date:\s*([\d-]+)', plan_text, re.DOTALL)
    duration = metadata.group(1) if metadata else ""
    exam_date = metadata.group(2) if metadata else ""
    
    # Extract objectives
    objectives = []
    objectives_section = re.search(r'Key Objectives:(.*?)(?=##|\Z)', plan_text, re.DOTALL)
    if objectives_section:
        for line in objectives_section.group(1).split('\n'):
            if line.strip().startswith('-'):
                objectives.append(line.strip()[1:].strip())
    
    # Extract daily schedule
    days = []
    day_sections = re.finditer(r'##\s*Day\s*(\d+):\s*([\d-]+)\s*\((\d+)\s*hours?\)(.*?)(?=##|\Z)', plan_text, re.DOTALL)
    
    for day in day_sections:
        activities = []
        for line in day.group(4).split('\n'):
            if line.strip().startswith('-'):
                activities.append(line.strip()[1:].strip())
        
        days.append({
            'number': day.group(1),
            'date': day.group(2),
            'hours': day.group(3),
            'activities': activities
        })
    
    return {
        'title': title,
        'duration': duration,
        'exam_date': exam_date,
        'objectives': objectives,
        'days': days
    }

def display_study_plan(plan_text):
    """Display the study plan in a formatted way."""
    plan = parse_study_plan(plan_text)
    
    # Display plan header
    st.markdown(f"""
    <div class="plan-header">
        <h2>{plan['title']}</h2>
        <p>Duration: {plan['duration']} days | Exam Date: {plan['exam_date']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display objectives
    if plan['objectives']:
        st.markdown('<div class="objectives">', unsafe_allow_html=True)
        st.markdown("### Key Objectives")
        for obj in plan['objectives']:
            st.markdown(f"• {obj}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display daily schedule
    st.markdown("### Daily Schedule")
    for day in plan['days']:
        st.markdown(f"""
        <div class="day-card">
            <div class="day-header">Day {day['number']}: {day['date']} ({day['hours']} hours)</div>
            {''.join(f'<div class="activity">• {activity}</div>' for activity in day['activities'])}
        </div>
        """, unsafe_allow_html=True)

def main():
    st.title("AI Study Buddy - Study Planner")
    st.write("Generate personalized study plans based on your lecture materials.")
    
    # Initialize the study plan generator
    generator = st.session_state.generator
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Study Plan Settings")
        
        # Get available subjects
        documents_dir = "documents"
        if not os.path.exists(documents_dir):
            st.error(f"Documents directory '{documents_dir}' not found!")
            return
        
        subjects = [d for d in os.listdir(documents_dir) 
                   if os.path.isdir(os.path.join(documents_dir, d)) and not d.startswith('.')]
        
        if not subjects:
            st.warning("No subject folders found in documents directory!")
            return
        
        # Subject selection
        selected_subject = st.selectbox(
            "Select Subject:",
            subjects,
            index=0 if subjects else None
        )
        
        if not selected_subject:
            return
        
        # Get PDFs for selected subject
        pdf_list = generator.get_pdf_list(selected_subject)
        
        if not pdf_list:
            st.info(f"No PDF files found for subject '{selected_subject}'")
            return
        
        # Target date selection
        target_date = st.date_input(
            "Target Completion Date:",
            value=datetime.date.today() + datetime.timedelta(days=7),
            min_value=datetime.date.today()
        )
        
        # PDF selection
        st.header("Select Lecture PDFs")
        selected_pdfs = st.multiselect(
            "Choose PDFs to include:",
            pdf_list
        )
        
        if selected_pdfs:
            # Generate plan button
            if st.button("Generate Study Plan", type="primary"):
                with st.spinner("Generating study plan..."):
                    plan = generator.generate_study_plan(
                        selected_subject,
                        selected_pdfs,
                        target_date.isoformat()
                    )
                    
                    if plan:
                        st.session_state.study_plan = plan
                        st.rerun()
    
    # Main content area
    if selected_pdfs:
        # PDF Summaries section
        st.header("PDF Summaries")
        if st.button("Preview PDF Summaries"):
            summaries = generator.ensure_summaries(selected_subject, selected_pdfs)
            
            for pdf_name, summary in summaries.items():
                with st.expander(f"Summary: {pdf_name}"):
                    if "TOPICS:" in summary and "SUMMARY:" in summary:
                        topics = summary.split("TOPICS:")[1].split("SUMMARY:")[0].strip()
                        overall_summary = summary.split("SUMMARY:")[1].strip()
                        
                        st.subheader("Topics and Concepts")
                        st.write(topics)
                        st.subheader("Overall Summary")
                        st.write(overall_summary)
                    else:
                        st.write(summary)
        
        # Study Plan section
        if hasattr(st.session_state, 'study_plan'):
            st.header("Generated Study Plan")
            display_study_plan(st.session_state.study_plan)
            
            # Download button
            st.download_button(
                label="Download Study Plan",
                data=st.session_state.study_plan,
                file_name=f"study_plan_{selected_subject}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Simple help section
    with st.expander("How it works"):
        st.write("""
        1. Select your subject and target completion date
        2. Choose the PDF lecture materials to include
        3. Preview the PDF summaries to see extracted topics
        4. Generate a personalized study plan
        5. Download the plan for offline use
        """)

if __name__ == "__main__":
    main()
