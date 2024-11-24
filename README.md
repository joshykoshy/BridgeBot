# BridgeBot

> A culturally sensitive, RAG-powered conversational AI built to answer sensitive cultural questions with empathy, factual accuracy, and historical nuance.

🥉 **3rd Place Winner** — **UOWD Tech Club AI Hackathon** (University of Wollongong in Dubai, November 2024)

---

## Overview

In globalized and multicultural communities, discussing cultural customs, religious traditions, and social etiquette often requires high sensitivity, historical context, and deep empathy. Standard conversational LLMs frequently stumble when handling sensitive cultural queries—either generalizing nuanced practices, generating subtle stereotypes, or failing to address taboo topics tactfully.

**BridgeBot** was developed during the UOWD Tech Club AI Hackathon to address this challenge. It utilizes a **Retrieval-Augmented Generation (RAG)** architecture that grounds large language model responses in verified cultural and religious history corpora. Through semantic document retrieval, custom prompt engineering, and a conversational interface, BridgeBot delivers respectful, informative, and empathetic answers to complex cultural inquiries.

---

## How It Works

BridgeBot executes a 4-stage RAG pipeline to ensure every response is factually grounded and culturally respectful:

```
┌─────────────────────────────────┐
│  Cultural Knowledge Sources     │
│  (Database of Religious History)│
└────────────────┬────────────────┘
                 │ WebBaseLoader + RecursiveCharacterTextSplitter
                 ▼
┌─────────────────────────────────┐
│     HuggingFace Embeddings      │
│  (all-mpnet-base-v2 Vector DB)  │
└────────────────┬────────────────┘
                 │ Semantic Similarity Retrieval
                 ▼
┌─────────────────────────────────┐      ┌─────────────────────────┐
│       Prompt Orchestration      │ ◄────┤   User Input / History  │
│  - Cultural empathy directives  │      └─────────────────────────┘
│  - Decorum & safety guardrails  │
│  - Injected context passages    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       OpenAI GPT-4o-mini        │
│  - Nuanced, empathetic response │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│        Gradio Web UI            │
│   (Interactive chat & feedback) │
└─────────────────────────────────┘
```

### The RAG Pipeline

1. **Document Ingestion & Semantic Chunking**:
   - Scrapes comprehensive cultural and religious history data using LangChain's `WebBaseLoader` (e.g., from *The Database of Religious History*).
   - Segments raw text into manageable units using `RecursiveCharacterTextSplitter` with a `chunk_size` of 200 characters and a `chunk_overlap` of 10 to preserve contextual continuity across boundaries.
2. **Dense Vector Embeddings & In-Memory Indexing**:
   - Generates high-dimensional semantic vector embeddings using HuggingFace's `sentence-transformers/all-mpnet-base-v2`.
   - Stores embeddings in an `InMemoryVectorStore` for low-latency k-NN semantic similarity retrieval.
3. **Context Injection & Prompt Engineering**:
   - Dynamically searches the vector space for the most relevant cultural passages matching the user's query.
   - Formulates a system prompt using a specialized `culture_prompt` template engineered to:
     - Clarify unfamiliar practices and correct misconceptions.
     - Maintain an open-minded and welcoming tone toward all cultures.
     - Respectfully de-escalate offensive inputs while explaining why decorum matters.
4. **Empathetic Generation & Multi-Turn State**:
   - Synthesizes retrieved context and user query via `ChatOpenAI` (`gpt-4o-mini`).
   - Appends user inputs and assistant answers to a persistent `chathistory` array to maintain multi-turn conversational context.

---

## Tech Stack

| Component | Technology / Framework |
|---|---|
| **LLM & Orchestration** | LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-huggingface`), OpenAI API (`gpt-4o-mini`) |
| **Embeddings & Search** | HuggingFace Sentence Transformers (`all-mpnet-base-v2`), PyTorch, LangChain `InMemoryVectorStore` |
| **Data Ingestion** | `WebBaseLoader`, BeautifulSoup4 |
| **User Interface** | Gradio 5.x |
| **Environment / Runtime** | Python 3.10+, `python-dotenv` |

---

## Setup & Installation

### Prerequisites

- **Python**: Version 3.10 or higher
- **OpenAI API Key**: Set with access to `gpt-4o-mini`

---

### 1. Clone the Repository & Setup Environment

```bash
git clone https://github.com/mayankt411/BridgeBot.git
cd BridgeBot

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

### 4. Run the Application

You can launch BridgeBot either via the standalone script or the Jupyter Notebook:

#### Option A: Gradio Web Interface
```bash
python app.py
```
*Open `http://127.0.0.1:7860` in your browser to start interacting.*

#### Option B: Jupyter Notebook
```bash
jupyter notebook Hackathon.ipynb
```

---

## Example Interaction

### Sample Query
> **User**: *"What is an appropriate gift to give a Brazilian at the Rio Carnival?"*

### Sample BridgeBot Response
> **BridgeBot**:
> *"When attending Rio Carnival, gift-giving is generally casual and festive rather than formal. If you are visiting friends or hosts during Carnival season, thoughtful and culturally appropriate gifts include:*
> 
> - * **Festive Accessories or Costumes**: Glitter, decorative headpieces, customized masks, or colorful street accessories for block parties (*blocos*).*
> - * **Hospitality Items**: A bottle of quality artisanal cachaça, premium Brazilian sweets (like brigadeiros), or refreshments to share with the group before heading to the Sambadrome.*
> - * **Practical Gifts**: Portable water bottles, sun protection, or cross-body bags for safety during crowded street celebrations.*
> 
> *In Brazilian culture, warmth, companionship, and shared celebration are valued above expensive material gifts. Avoid overly solemn or business-like gifts during Carnival, as the festival is focused on expression, community, and joy."*

---

## Award & Recognition

- 🥉 **3rd Place Winner** — **UOWD Tech Club AI Hackathon** (University of Wollongong in Dubai, November 2024).
- Awarded for pioneering an empathetic, retrieval-grounded conversational agent designed to bridge cultural divides and educate users with accuracy.

---

## License

This project is licensed under the [MIT License](LICENSE).
