import streamlit as st
from document_utils import get_subjects, get_pdfs_for_subject, read_pdf, get_pdf_pages
from quiz_generator import QuizMaker

# Page config
st.set_page_config(page_title="Study Buddy Quiz", page_icon="📚")
st.title("Study Buddy Quiz Generator 📚")

# Initialize quiz maker
quiz_maker = QuizMaker()

# Initialize session state
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "results" not in st.session_state:
    st.session_state.results = []
if "quiz_complete" not in st.session_state:
    st.session_state.quiz_complete = False

# Sidebar for document selection and quiz settings
with st.sidebar:
    st.header("Quiz Settings")
    
    # Get available subjects
    subjects = get_subjects()
    if not subjects:
        st.error("No subjects found in documents folder!")
        st.stop()
        
    # Subject selection
    subject = st.selectbox("Select Subject:", subjects)
    
    # Get PDFs for selected subject
    pdfs = get_pdfs_for_subject(subject)
    if not pdfs:
        st.error(f"No PDFs found for {subject}!")
        st.stop()
        
    # PDF selection
    pdf_file = st.selectbox("Select PDF:", pdfs)
    
    # Show PDF page count
    if pdf_file:
        try:
            pages = get_pdf_pages(subject, pdf_file)
            st.info(f"📄 {pages} pages")
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            st.stop()
    
    # Quiz options
    difficulty = st.select_slider(
        "Difficulty:",
        options=["easy", "medium", "hard"],
        value="medium"
    )
    
    num_questions = st.slider(
        "Number of Questions:",
        min_value=5,
        max_value=15,
        value=5,
        step=5
    )
    
    # Generate quiz button
    if st.button("Generate Quiz"):
        with st.spinner("Creating your quiz..."):
            try:
                # Read PDF content
                content = read_pdf(subject, pdf_file)
                
                # Generate quiz
                quiz_data = quiz_maker.make_quiz(
                    content=content,
                    num_questions=num_questions,
                    difficulty=difficulty
                )
                
                # Reset quiz state
                st.session_state.quiz_data = quiz_data
                st.session_state.current_question = 0
                st.session_state.results = []
                st.session_state.quiz_complete = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to generate quiz: {str(e)}")

# Main quiz interface
if st.session_state.quiz_data:
    questions = st.session_state.quiz_data["questions"]
    current_q = st.session_state.current_question
    
    # Show progress
    progress = (current_q + 1) / len(questions)
    st.progress(progress)
    st.write(f"Question {current_q + 1} of {len(questions)}")
    
    # Display current question
    question = questions[current_q]
    st.write("### " + question["question"])
    
    # Display options and get answer
    col1, col2 = st.columns(2)
    with col1:
        for opt in ["A", "B"]:
            st.write(f"{opt}. {question['options'][opt]}")
    with col2:
        for opt in ["C", "D"]:
            st.write(f"{opt}. {question['options'][opt]}")
            
    # Get user's answer
    answer = st.radio("Your answer:", ["A", "B", "C", "D"], key=f"q_{current_q}")
    
    if st.button("Submit Answer"):
        # Check answer
        is_correct, feedback = quiz_maker.check_answer(question, answer)
        
        # Store result with the selected answer
        st.session_state.results.append({
            "question": current_q + 1,
            "correct": is_correct,
            "selected": answer
        })
        
        # Show feedback
        st.write("---")
        st.write(feedback)
        
        # Move to next question or finish
        if current_q < len(questions) - 1:
            st.session_state.current_question += 1
            st.rerun()
        else:
            st.session_state.quiz_complete = True
            st.rerun()

# Show final score and detailed summary
if st.session_state.quiz_complete:
    st.write("---")
    st.write("## Quiz Complete!")
    
    # Show overall score
    feedback = quiz_maker.calculate_score(st.session_state.results)
    st.write(feedback)
    
    # Show detailed question summary
    st.write("### Question Summary")
    questions = st.session_state.quiz_data["questions"]
    
    for i, result in enumerate(st.session_state.results):
        question = questions[i]
        is_correct = result["correct"]
        
        # Create expandable section for each question
        with st.expander(f"Question {i+1} - {'✅ Correct' if is_correct else '❌ Wrong'}"):
            st.write(f"**Q: {question['question']}**")
            
            # Show all options, highlight correct and selected answers
            selected = result["selected"]
            for opt in ["A", "B", "C", "D"]:
                if opt == question["correct"]:
                    st.success(f"{opt}. {question['options'][opt]} (Correct Answer)")
                elif opt == selected and not is_correct:
                    st.error(f"{opt}. {question['options'][opt]} (Your Answer)")
                else:
                    st.write(f"{opt}. {question['options'][opt]}")
            
            # Show explanation
            st.write("---")
            st.write("**Explanation:**")
            st.write(question["explanation"])
    
    if st.button("Start New Quiz"):
        # Reset everything
        st.session_state.quiz_data = None
        st.session_state.current_question = 0
        st.session_state.results = []
        st.session_state.quiz_complete = False
        st.rerun()
