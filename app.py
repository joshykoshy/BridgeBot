import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment. Please set it in a .env file.")

print("Loading cultural knowledge base...")
loader = WebBaseLoader("https://religiondatabase.org/browse")
raw_documents = loader.load()

# Split documents into semantic chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
documents = splitter.split_documents(raw_documents)
print(f"Indexed {len(documents)} document chunks.")

# Initialize embeddings & in-memory vector store
print("Initializing sentence embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore.from_documents(documents, embeddings)

# Initialize OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# Prompt template emphasizing cultural sensitivity, nuance, and respect
culture_template = """
You are an AI system focused on cultural heritage. You understand diverse cultural practices, historical contexts, and social nuances. 
If you encounter a topic you're unfamiliar with, ask for more details. 
Correct any misinformation and provide culturally sensitive, factual information. 
When users ask about cultural practices, share insights and historical context.
Ask to maintain decorum and respect in all interactions, if user's prompt is offensive, and explain why it's inappropriate.
Maintain an open-minded view for all cultures and practices.

Current chat history: {chat_history}
Relevant context: {context}

User Input: {user_input}

Provide a detailed and respectful response:
"""

culture_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "user_input"],
    template=culture_template
)

chathistory = []

def generate_response(user_input):
    if not user_input.strip():
        return "Please enter a question or topic about cultural practices."
    
    # Retrieve relevant context via semantic similarity search
    docs = [doc.page_content for doc in vector_store.similarity_search(user_input)]
    
    # Generate culturally empathetic response
    formatted_prompt = culture_prompt.format(
        chat_history="\n".join(chathistory[-5:]),
        context="\n".join(docs),
        user_input=user_input
    )
    
    response = llm.invoke(formatted_prompt).content
    
    # Update multi-turn history
    chathistory.append(f"User: {user_input}")
    chathistory.append(f"BridgeBot: {response}")
    
    return response

# Create and launch Gradio interface
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about cultural etiquette, religious traditions, or sensitive practices..."),
    outputs=gr.Textbox(lines=10, label="BridgeBot Response"),
    title="BridgeBot — Cultural Heritage & Sensitivity AI",
    description="A RAG-powered conversational assistant built to answer sensitive cultural and religious questions with empathy, nuance, and historical accuracy.",
    examples=[
        ["What is an appropriate gift to give a Brazilian at the Rio Carnival?"],
        ["What are key etiquette guidelines when attending a traditional Japanese tea ceremony?"],
        ["How should one dress respectfully when visiting a mosque or temple?"]
    ]
)

if __name__ == "__main__":
    interface.launch()
