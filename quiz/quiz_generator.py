from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

class QuizMaker:
    def __init__(self):
        self.client = OpenAI()
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def make_quiz(self, content, num_questions=5, difficulty="medium"):
        """Create a quiz from the given content"""
        tokens = self.encoding.encode(content)
        total_tokens = len(tokens)
        
        # For small content, use it all
        if total_tokens <= 10000:
            return self.generate_questions(content, num_questions, difficulty)
            
        # For larger content, split into 10k chunks
        questions = []
        chunk_size = 10000
        
        for i in range(0, total_tokens, chunk_size):
            # Get chunk of 10k tokens
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate remaining questions needed
            remaining = num_questions - len(questions)
            if remaining <= 0:
                break
                
            # Generate questions from this chunk
            chunk_quiz = self.generate_questions(chunk_text, remaining, difficulty)
            questions.extend(chunk_quiz["questions"])
        
        return {"questions": questions}
    
    def generate_questions(self, content, num_questions, difficulty):
        """Generate questions from content"""
        prompt = f"""
        Create {num_questions} multiple choice questions from this content. 
        Difficulty level: {difficulty}
        
        Content:
        {content}
        
        Make questions that test understanding, not just memorization.
        For each question:
        - Include 4 options (A, B, C, D)
        - Mark the correct answer
        - Add a brief explanation why it's correct
        
        Return as JSON with this structure:
        {{
            "questions": [
                {{
                    "question": "...",
                    "options": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }},
                    "correct": "A/B/C/D",
                    "explanation": "..."
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful teacher creating quiz questions."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise Exception(f"Couldn't generate quiz: {str(e)}")
    
    def check_answer(self, question_data, student_answer):
        """Check if the answer is correct and return feedback"""
        is_correct = student_answer.upper() == question_data["correct"]
        
        if is_correct:
            feedback = "✅ Correct! " + question_data["explanation"]
        else:
            feedback = f"❌ Not quite. The correct answer was {question_data['correct']}. "
            feedback += question_data["explanation"]
            
        return is_correct, feedback
    
    def calculate_score(self, results):
        """Calculate the final score and return feedback"""
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        score = (correct / total) * 100
        
        feedback = f"You got {correct} out of {total} questions correct ({score:.1f}%).\n\n"
        
        # Add some encouraging feedback
        if score == 100:
            feedback += "Perfect score! Amazing work! 🌟"
        elif score >= 80:
            feedback += "Great job! You've got a solid understanding! 👏"
        elif score >= 60:
            feedback += "Good effort! Keep practicing to improve! 💪"
        else:
            feedback += "Keep studying! You'll get there! 📚"
            
        return feedback
