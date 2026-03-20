
Overview
--------
This project implements **Retrieval-Augmented Generation (RAG)** pipeline that retrieves course-specific materials from a MongoDB Atlas vector store and uses an OpenAI model to generate context-aware answers. The system leverages multi-query transformations and RAG-fusion to enhance retrieval,and employs LangChain  with LLM integration for seamless response generation.Together, these components enable intelligent book recommendations and question answering that are both precise and contextually relevant. The end product of this project is a streamlit app that recommends books based on user query

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)

Key features
- Query transformation for higher-quality retrieval  
- External prompt files (`prompts/`) for maintainability  
- Conda-based environment reproducibility  
- LangSmith traces for debugging and transparency
- Streamlit app in `rag_app/` and supporting notebooks for exploration



## Repository Structure
```
Group-1/
├── .env.example                          # Template                          
├── environment.yml                       # environment specification
├── notebooks/                            # Jupyter notebooks     
│   └── rag.ipynb                         # RAG pipeline 
├── prompts/                              # Prompt templates
│   ├── README.md                         # Prompts documentation
│   ├── get_response_v1.txt               # Response template
│   ├── multi_query_v1.txt                # Multi-query expansion
│   └── system_prompt_v1.txt              # System instructions
├── rag_app/                        # Streamlit RAG application
│   ├── application.py                    # Core RAG logic
│   └── requirements.txt                  # App dependencies
└── README.md
    
```
---

## Data Preparation

### Clone Repository
```bash
git clone https://github.com/EstherAbik/Librarian.git
cd Librarian
```

### 1️⃣ Acquire Dataset
Download the dataset from Kaggle: [Goodreads Book Reviews Dataset](https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1?select=goodreads_reviews_dedup.json)

**Collections used:**
- `reviews` - Book reviews data
- `books` - Book metadata
- `authors` - Author information

### 2️⃣ Filter Dataset
Restrict reviews to those with `date_started` in **August 2017** and match with corresponding books.

### 3️⃣ Configure MongoDB Credentials
Create a file named `credentials_mongodb.json` in the `notebooks/` directory with the following template:

```json
{
    "host": "your-mongodb-host",
    "username": "your-username",
    "password": "your-password"
}
```

⚠️ **Important:** Do not commit this file to GitHub. It is excluded via `.gitignore` for security.

### 4️⃣ Upload Data to MongoDB

### 5️⃣ Perform Schema Transformation

---

## ⚙️ Environment Setup 

### 1️⃣ Create Conda Environment
```bash
conda env create -f environment.yml
conda activate rag-pipeline
```

### 2️⃣ Verify Python and Package Versions
```bash
python --version       # expected: 3.12.12
conda list langchain    # check version matches environment.yml
```

### 3️⃣ Set Up Environment Variables
Copy `.env.example` → `.env` and fill in your own credentials:
```
OPENAI_API_KEY=sk-xxxx
MONGODB_URI=your_mongo_connection_uri
```

Ensure `.env` is **not** committed (`.gitignore` includes it).



## 4️⃣ Set up Streamlit secrets

Create a `.streamlit/secrets.toml` file with the following structure and **replace your API keys** accordingly:

```toml
[langsmith]
tracing = "true"
endpoint = "https://api.smith.langchain.com"
api_key = "your-langsmith-api-key"

[google]
api_key = "your-google-api-key"

[mongodb]
uri = "mongodb+srv://user1:user1@cluster0.lqirl.mongodb.net/?retryWrites=true&w=majority"
```
---

## 🚀 Running the Pipeline

### Launch Jupyter Notebooks
```bash
jupyter notebook notebooks/rag.ipynb
```

---

## 💬 Example Query Walkthrough
**Query:**  
> “I'm looking for a book to help me improve my leadership skills and communicate more effectively at work”

**Under the hood:**
- Multi‑query transformation:
  - The original query is expanded into 4 semantically diverse sub‑queries, :
    - Question 1: Best books for improving leadership and communication skills
    - Question 2: Books on effective workplace communication and leadership development
    - Question 3: Top books for enhancing leadership and communication in a professional setting
    - Question 4: Recommended books for managers to improve communication and leadership abilities

- RAG‑Fusion step:
  - Each sub‑query is sent to MongoDB Atlas Vector Search.
  - Top documents from each query are retrieved.
  - Results are fused (deduplicated, ranked) into a final set of top‑k documents.

- Prompts combined:
  - system_prompt_v1
  - get_response_v1
  - multi_query_v1

- LLM:
  - gpt‑4o‑mini

- Expected Output (excerpt):
  - "'Based on the context, "The Coaching Habit: Say Less, Ask More & Change the Way You Lead Forever" seems like an excellent fit for your needs. One review describes it as a "Great book on conversations: how to listen more and ask the right questions," which directly addresses improving communication skills. Its title also indicates a focus on changing how you lead...'"

---

### Run the Streamlit Application
```bash
cd rag_app
streamlit run application.py
```

Access the app at `http://localhost:8501`

---

## 🤖 RAG Application Features

### Architecture
- **Frontend**: Streamlit with custom neobrutalist CSS styling
- **Vector Database**: MongoDB Atlas with vector search capabilities  
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM**: Google Gemini 2.5 Flash for response generation
- **Framework**: LangChain for RAG pipeline orchestration
- **Monitoring**: LangSmith for tracing and debugging

### Key Features
- 🔍 **Multi-Query Retrieval**: Expands user queries into multiple perspectives for better recall
- 🎯 **RAG Fusion**: Re-ranks retrieved documents by frequency scoring
- 💬 **Streaming Responses**: Real-time AI response generation
- 📊 **Session Tracking**: Monitor conversation metrics in the sidebar
- 🎨 **Modern UI**: Bold, colorful design with custom CSS
- 🔗 **Context-Aware**: Maintains chat history for coherent conversations

### RAG Pipeline Workflow
1. **Query Transformation**: User query expanded into 4 semantically diverse sub-queries
2. **Vector Search**: Each sub-query retrieves relevant documents from MongoDB Atlas
3. **RAG Fusion**: Results are fused and re-ranked by document frequency
4. **Context Building**: Top documents formatted with book titles, reviews, and ratings
5. **LLM Generation**: Gemini AI generates response using retrieved context and chat history
6. **Streaming Output**: Response streamed back to user interface

### Customization Options

**Modify AI Personality**  
Edit prompt templates in `prompts/` directory:
- `system_prompt_v1.txt` - Define behavior and tone
- `get_response_v1.txt` - Response format and rules  
- `multi_query_v1.txt` - Query expansion instructions

**Adjust Vector Search**  
Configure retriever in `application.py`:
```python
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="realvector_index",
    relevance_score_fn="cosine"  # or "euclidean", "dotProduct"
)
```

**Change Styling**  
Modify custom CSS variables in `application.py` for colors and design.

### Monitoring & Debugging
- **LangSmith Tracing**: View all LLM calls, retrieval results, and performance metrics
- **Session Metrics**: Track message count and conversation flow in sidebar
- **Error Handling**: Graceful fallbacks for API failures and network issues

### Troubleshooting
- **Vector Search Issues**: Ensure index dimension matches embedding model (384 for all-MiniLM-L6-v2)
- **Connection Errors**: Verify MongoDB URI and API keys in `.streamlit/secrets.toml`
- **Slow Responses**: Check network latency, optimize retriever parameters, or use closer MongoDB region



## 🧾 Prompts and Documentation
All prompts are stored and versioned under `/prompts`.

| File | Purpose | Version |
|------|----------|----------|
| `system_prompt_v1.txt` | Defines assistant behavior and tone | v1.0 |
| `get_response_v1.txt` | Template for user query + context insertion | v1.0 |
| `multi_query_v1.txt`. | Expands the original question into multiple newline‑separated queries to increase recall. | v1.0 |

Example loader:
```python
from utils import load_prompt
system_prompt = load_prompt(".../prompts/system_prompt_v1.txt")
human_prompt = load_prompt(".../prompts/get_response_v1.txt")
multi_query_prompt = load_prompt(".../prompts/multi_query_v1.txt")
```
See `prompts/README.md` for variable placeholders like `{query}` and `{context}`.


---

## 🧭 Reflection
**Challenges:** We encountered limitations with token capacity, which led us to adopt a more flexible model architecture without strict token restrictions.

---

 



