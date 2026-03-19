"""
MongoDB Vector Search Chatbot with LangChain and Streamlit
"""

# Standard library imports
import os
from operator import itemgetter

# Third-party imports
import streamlit as st
from pymongo import MongoClient

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.load import dumps, loads
from collections import Counter


# Load environment variables
# LangSmith / LangChain
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["langsmith"]["tracing"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["langsmith"]["endpoint"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["langsmith"]["api_key"]

# Google
os.environ["GOOGLE_API_KEY"] = st.secrets["google"]["api_key"]

# Mongo
os.environ["MONGO_URI"] = st.secrets["mongodb"]["uri"]


def load_prompt(rel_path: str) -> str:
    """Load a prompt file from the `prompts/` directory located next to this file.

    Returns the file contents as a string. If loading fails, returns an empty string.
    """
    try:
        base = os.path.dirname(__file__)
        path = os.path.join(base, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        try:
            st.error(f"Failed to load prompt {rel_path}: {e}")
        except Exception:
            pass
        return ""

@st.cache_resource
def get_mongodb_collection():
    """Connect to MongoDB and return collection (cached)."""
    MONGODB_URI = os.environ['MONGO_URI']
    client = MongoClient(MONGODB_URI)
    database = client['goodreads'] #replace it with your own database
    return database['reviews'] #replace it with your own collection


from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

@st.cache_resource
def get_embeddings_and_llm():
    """Initialize embeddings and LLM models (cached)."""
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    llm = init_chat_model(model="gemini-2.5-flash",streaming=True)
    return embeddings, llm

@st.cache_resource
def get_embeddings_and_llm():
    """Initialize embeddings and LLM models (cached)."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    return embeddings, llm

@st.cache_resource
def get_retriever():
    """Initialize vector store retriever (cached)."""
    collection = get_mongodb_collection()
    embeddings, _ = get_embeddings_and_llm()
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        text_key="review_text", #replace this with your field name of the reviews
        index_name="realvector_index", #replace it with your own
        relevance_score_fn="cosine"
    )
    return vector_store.as_retriever()

# Initialize cached resources
collection = get_mongodb_collection()
embeddings, llm = get_embeddings_and_llm()
retriever = get_retriever()

# Helper function
def format_docs(docs, context="summary"):
    """
    Format retrieved Goodreads documents into a string based on context.
    Fields available:
      - _id: internal identifier
      - book_id: Goodreads book identifier
      - user_id: reviewer identifier
      - book_title: title of the book
      - review_text: text of the review
      - rating: numeric rating (1–5)
      - date_added: when the review was added

    Context options:
      - "reviews": show review_text(s)
      - "ratings": show rating if single, average if multiple
      - "summary": show title + summary (default)
      - "metadata": show all fields
    """
    parts = []
    ratings = []

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        # Support dict-like docs too
        get = lambda key: metadata.get(key) if metadata else doc.get(key) if isinstance(doc, dict) else None

        _id = get("_id")
        book_id = get("book_id")
        user_id = get("user_id")
        book_title = get("book_title") or "Unknown title"
        review_text = get("review_text") or "No reviews available"
        rating = get("rating")
        date_added = get("date_added") or "Unknown date"

        content = doc.page_content if hasattr(doc, "page_content") else str(doc)

        if isinstance(rating, (int, float)):
            ratings.append(rating)

        entry = [f"Title: {book_title}"]

        if context == "summary":
            entry.append(f"Summary: {content}")
        elif context == "reviews":
            entry.append(f"Review: {review_text}")
        elif context == "ratings" and rating is not None:
            entry.append(f"Rating: {rating}")
        elif context == "metadata":
            entry.extend([
                f"ID: {_id}",
                f"Book ID: {book_id}",
                f"User ID: {user_id}",
                f"Review: {review_text}",
                f"Rating: {rating if rating is not None else 'No rating'}",
                f"Date Added: {date_added}",
                f"Summary: {content}"
            ])

        parts.append("\n".join(entry))

    # Ratings summary logic
    rating_summary = ""
    if context == "ratings":
        if len(ratings) > 1:
            avg_rating = sum(ratings) / len(ratings)
            rating_summary = f"\n\nAverage Rating (across {len(ratings)} reviews): {avg_rating:.2f}"
        elif len(ratings) == 1:
            rating_summary = f"\n\nOverall Rating: {ratings[0]:.2f}"

    return "\n\n".join(parts) + rating_summary




    
# Streamlit app config
st.set_page_config(
    page_title="AI Chat Assistant 2025",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    :root {
        --ink: #111111;
        --bg: #f7f7f7;

        /* AI messages: now gold */
        --ai: #FFD700;
        --ai-ink: #B8860B;

        /* User messages */
        --user: #FFF0D6;
        --user-ink: #FF8A00;

        /* Accent colors: blue with silver */
        --accent: #1E90FF;
        --accent-2: #C0C0C0;
    }

    /* App background with image */
    .stApp {
        background: url("https://img.freepik.com/premium-photo/libery-books-ai-generator_1026664-172.jpg?w=2000") no-repeat center center fixed;
        background-size: cover;
    }

    /* Chat messages */
    .stChatMessage {
        background: #ffffff !important;
        border: 3px solid var(--ink);
        border-radius: 14px;
        padding: 1rem;
        margin: 0.75rem 0;
        box-shadow: 8px 8px 0 var(--ink);
    }

    /* AI and Human variations */
    .stChatMessage[data-testid*="ai"] {
        background: var(--ai) !important;
        border-color: var(--ai-ink);
        box-shadow: 8px 8px 0 var(--ai-ink);
    }
    
    .stChatMessage[data-testid*="user"] {
        background: var(--user) !important;
        border-color: var(--user-ink);
        box-shadow: 8px 8px 0 var(--user-ink);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--accent);
        border-left: 3px solid var(--ink);
        box-shadow: -8px 0 0 var(--accent-2);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    /* Buttons */
    .stButton button {
        background: var(--accent-2) !important;
        color: var(--ink) !important;
        border: 3px solid var(--ink) !important;
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--ink);
        transition: transform 0.1s ease, box-shadow 0.1s ease;
    }
    
    .stButton button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0 var(--accent);
    }

    /* Titles */
    h1 {
        color: var(--ink);
        text-align: left;
        font-weight: 900;
        display: inline-block;
        background: var(--accent);
        padding: 6px 12px;
        border: 3px solid var(--ink);
        border-radius: 12px;
        box-shadow: 6px 6px 0 var(--accent-2);
    }

    /* Make h3 white (for "Ask me anything!") */
    h3 {
        color: #ffffff !important;
        text-align: left;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)



# Sidebar
with st.sidebar:
    st.markdown("### 🤖 AI Chat Assistant")
    st.markdown("---")
    
    st.markdown("#### 📊 Session Info")
    if "chat_history" in st.session_state:
        msg_count = len([m for m in st.session_state.chat_history if isinstance(m, HumanMessage)])
        st.metric("Messages Sent", msg_count)
    
    st.markdown("---")
    st.markdown("#### ⚙️ Settings")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your AI assistant. How can I help you today?")
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("#### 💡 About")
    st.markdown("""
    This chatbot uses:
    - 🔍 MongoDB Atlas Vector Search
    - 🤖 Google Gemini AI
    - 🔗 LangChain RAG
    - 📊 LangSmith Tracing
    """)
    
    st.markdown("---")
    st.markdown("##### Made with ❤️ using Streamlit")

# Main header
st.title("🤖 Librarian")
st.markdown("### Ask me anything!")
st.markdown("---")

# --- Multi-query prompt (loaded from prompts/) ---
multi_query_template = load_prompt("../prompts/multi_query_v1.txt")

prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

# --- Split queries into list ---
split_queries = RunnableLambda(lambda x: x.split("\n"))

# --- RAG Fusion: re-rank by frequency ---
def rag_fusion(docs_per_query):
    """Fuse multi-query retrieval results with frequency scoring."""
    all_docs = [dumps(doc) for sublist in docs_per_query for doc in sublist]
    counts = Counter(all_docs)
    # Sort by frequency (most common first)
    ranked_docs = [loads(doc) for doc, _ in counts.most_common()]
    return ranked_docs

rag_fusion_runnable = RunnableLambda(rag_fusion)

# --- Build multi-query + fusion chain ---
generate_queries = prompt_perspectives | llm | StrOutputParser() | split_queries

multi_query_retrieval_chain = (
    generate_queries
    | retriever.map()      # run retriever for each query
    | rag_fusion_runnable  # fuse results by frequency
    | format_docs          # format into readable context
)

def get_response(user_query, chat_history):
    # Load main prompt from prompts/get_response_v1.txt (versioned files, not hard-coded)
    template = load_prompt("../prompts/get_response_v1.txt")

    # Load system prompt and prepend (system instructions are kept in prompts/system_prompt_v1.txt)
    system_prompt = load_prompt("../prompts/system_prompt_v1.txt")
    template = system_prompt + "\n\n" + template

    # Build the prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Chain definition
    chain = (
        {
            # Instead of a single retriever, use the multi-query retrieval chain
            "context": itemgetter("question") | multi_query_retrieval_chain,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Stream results back to Streamlit
    return chain.stream({
        "chat_history": chat_history,
        "question": user_query,
    })


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, welcome to goodreads?"),
    ]

# Display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))





