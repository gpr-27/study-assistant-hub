from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

class FlashcardMaker:
    def __init__(self):
        self.client = OpenAI()
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def make_flashcards(self, content, num_cards=10, difficulty="medium"):
        """Create flashcards from the given content"""
        tokens = self.encoding.encode(content)
        total_tokens = len(tokens)
        
        # For small content, use it all
        if total_tokens <= 10000:
            return self.generate_flashcards(content, num_cards, difficulty)
            
        # For larger content, split into 10k chunks
        cards = []
        chunk_size = 10000
        
        for i in range(0, total_tokens, chunk_size):
            # Get chunk of 10k tokens
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate remaining cards needed
            remaining = num_cards - len(cards)
            if remaining <= 0:
                break
                
            # Generate flashcards from this chunk
            chunk_cards = self.generate_flashcards(chunk_text, remaining, difficulty)
            cards.extend(chunk_cards["cards"])
        
        return {"cards": cards}
    
    def generate_flashcards(self, content, num_cards, difficulty):
        """Generate flashcards from content"""
        prompt = f"""
        Create {num_cards} focused flashcards from this content.
        Difficulty level: {difficulty}

        Rules for creating effective flashcards:
        1. One concept per card - break complex ideas into multiple cards
        2. Front should require active recall, not just recognition
        3. Questions should be specific and clear
        4. Answers should be concise but complete
        5. Avoid true/false or multiple choice formats
        6. For {difficulty} difficulty, choose simpler/harder concepts, but keep cards clear

        Examples of good flashcards:
        - Front: "What is Newton's First Law of Motion?"
          Back: "An object at rest stays at rest, and an object in motion stays in motion, unless acted upon by an external force."
          Topic: "Physics - Motion"

        - Front: "Complete the equation: E = ___"
          Back: "mc²"
          Topic: "Physics - Energy"

        Content:
        {content}
        
        Return as JSON with this structure:
        {{
            "cards": [
                {{
                    "front": "Question or term goes here",
                    "back": "Answer or definition goes here",
                    "topic": "Brief topic/category label"
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful teacher creating educational flashcards."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise Exception(f"Couldn't generate flashcards: {str(e)}")
    

