
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

def create_qa_chain(retriever):
    """
    Sets up a chatbot that can answer questions about our docs.
    """
   
    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
    You are a skilled and thoughtful teaching assistant. Your job is to help students deeply understand course material by guiding them through ideas—not just giving answers.

    Please follow these guidelines:

    1. Only use the provided *context* to answer the question. Do **not** include outside information.
    2. Keep responses **clear, concise, and focused**—avoid lengthy explanations.
    3. Break down complex concepts into simpler parts when needed.
    4. Use analogies or examples if they help with understanding.
    5. Briefly connect to previous concepts if the material builds on them.
    6. If something is unclear or not supported by the context, say so.
    7. Encourage critical thinking through your explanation—don’t just provide a direct answer.

    ---

    **Course Material (Context):**  
    {context}

    **Previous Discussion:**  
    {chat_history}

    **Student’s Question:**  
    {question}

    ---

    **Your Response (concise, clear, and educational):**
    """
    )

    # this helps maintain memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = ChatOpenAI(model = "gpt-4o")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    return qa_chain, memory 
