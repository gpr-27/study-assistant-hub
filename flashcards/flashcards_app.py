import streamlit as st
from document_utils import get_subjects, get_pdfs_for_subject, read_pdf, get_pdf_pages
from flashcards_generator import FlashcardMaker
import json
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Study Buddy Flashcards", page_icon="🎴")
st.title("Study Buddy Flashcard Generator 🎴")

# Initialize flashcard maker
flashcard_maker = FlashcardMaker()

# Initialize session state
if "current_card" not in st.session_state:
    st.session_state.current_card = 0
if "cards_data" not in st.session_state:
    st.session_state.cards_data = None
if "show_back" not in st.session_state:
    st.session_state.show_back = False
if "card_knowledge" not in st.session_state:
    st.session_state.card_knowledge = {}  # Stores knowledge level for each card

# Sidebar for document selection and flashcard settings
with st.sidebar:
    st.header("Flashcard Settings")
    
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
    
    # Flashcard options
    difficulty = st.select_slider(
        "Difficulty:",
        options=["easy", "medium", "hard"],
        value="medium"
    )
    
    num_cards = st.slider(
        "Number of Cards:",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )
    
    # Study mode selection - only show review option if there are cards
    study_options = ["Learn New Cards"]
    if st.session_state.cards_data:  # Only add review option if cards exist
        study_options.append("Review Previous Cards")
    
    mode = st.radio("Study Mode:", study_options)
    review_mode = (mode == "Review Previous Cards")
    
    # Generate flashcards button - only show in Learn New Cards mode
    if mode == "Learn New Cards":
        if st.button("Generate Flashcards"):
            with st.spinner("Creating your flashcards..."):
                try:
                    # Read PDF content
                    content = read_pdf(subject, pdf_file)
                    
                    # Generate flashcards
                    cards_data = flashcard_maker.make_flashcards(
                        content=content,
                        num_cards=num_cards,
                        difficulty=difficulty
                    )
                    
                    # Reset flashcard state
                    st.session_state.cards_data = cards_data
                    st.session_state.current_card = 0
                    st.session_state.show_back = False
                    st.session_state.card_knowledge = {}
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to generate flashcards: {str(e)}")

# Main flashcard interface
if st.session_state.cards_data:
    cards = st.session_state.cards_data["cards"]
    
    # Handle review mode
    if review_mode:
        # Get cards that need review (not known well)
        review_cards = []
        for i, card in enumerate(cards):
            knowledge = st.session_state.card_knowledge.get(i, "new")
            if knowledge != "knew_well":
                # Sort by knowledge level - unknown first, then somewhat known
                priority = 0 if knowledge == "didnt_know" else 1 if knowledge == "somewhat_knew" else 2
                review_cards.append((priority, i, card))
        
        if not review_cards:
            st.info("No cards to review! You knew all cards well.")
            st.stop()
            
        # Sort by priority (unknown first, then somewhat known)
        review_cards.sort()
        
        # Use review cards instead of all cards
        current_idx = st.session_state.current_card % len(review_cards)
        _, card_idx, card = review_cards[current_idx]
        progress = (current_idx + 1) / len(review_cards)
        total = len(review_cards)
    else:
        # Normal mode - use all cards
        current_idx = st.session_state.current_card % len(cards)
        card_idx = current_idx
        card = cards[current_idx]
        progress = (current_idx + 1) / len(cards)
        total = len(cards)
    
    # Progress indicator
    st.caption(f"Card {current_idx + 1} of {total}")
    st.progress(progress)
    
    # Card display
    card_container = st.container(border=True)
    with card_container:
        st.caption(f"{card['topic']}")
        
        # Card content
        st.write("")  # Spacing
        if not st.session_state.show_back:
            st.header(card['front'])
        else:
            st.header(card['back'])
        st.write("")  # Spacing
        
        # Flip button
        cols = st.columns([2, 1, 2])
        with cols[1]:
            if st.button("🔄 Flip", use_container_width=True):
                st.session_state.show_back = not st.session_state.show_back
                st.rerun()
    
    # Navigation and assessment
    cols = st.columns(5)
    
    with cols[0]:
        if st.button("⬅️", use_container_width=True):
            st.session_state.current_card = (current_idx - 1) % total
            st.session_state.show_back = False
            st.rerun()
            
    with cols[1]:
        if st.button("❌", use_container_width=True, help="Didn't know"):
            st.session_state.card_knowledge[card_idx] = "didnt_know"
            st.session_state.current_card = (current_idx + 1) % total
            st.session_state.show_back = False
            st.rerun()
            
    with cols[2]:
        if st.button("⭐", use_container_width=True, help="Somewhat knew"):
            st.session_state.card_knowledge[card_idx] = "somewhat_knew"
            st.session_state.current_card = (current_idx + 1) % total
            st.session_state.show_back = False
            st.rerun()
            
    with cols[3]:
        if st.button("✅", use_container_width=True, help="Knew well"):
            st.session_state.card_knowledge[card_idx] = "knew_well"
            st.session_state.current_card = (current_idx + 1) % total
            st.session_state.show_back = False
            st.rerun()
            
    with cols[4]:
        if st.button("➡️", use_container_width=True):
            st.session_state.current_card = (current_idx + 1) % total
            st.session_state.show_back = False
            st.rerun()
    
    # Show statistics
    if st.session_state.card_knowledge:
        st.write("---")
        st.write("### Your Progress")
        
        # Calculate stats
        total_rated = len(st.session_state.card_knowledge)
        knew_well = sum(1 for r in st.session_state.card_knowledge.values() if r == "knew_well")
        somewhat = sum(1 for r in st.session_state.card_knowledge.values() if r == "somewhat_knew")
        didnt_know = sum(1 for r in st.session_state.card_knowledge.values() if r == "didnt_know")
        
        # Show stats in columns
        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.metric("Knew Well", f"{knew_well} ({knew_well/total_rated*100:.0f}%)")
        with stat_cols[1]:
            st.metric("Somewhat Knew", f"{somewhat} ({somewhat/total_rated*100:.0f}%)")
        with stat_cols[2]:
            st.metric("Didn't Know", f"{didnt_know} ({didnt_know/total_rated*100:.0f}%)")
