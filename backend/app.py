from fastapi import FastAPI, Request
import logging
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFacePipeline
import tempfile
from langchain.agents import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
# LangGraph imports
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("philomena_cunk")
load_dotenv()



# Redis cache for common questions
try:
    import redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )
    # Test connection
    try:
        redis_client.ping()
        logger.info("Connected to Redis server.")
    except Exception as e:
        logger.warning(f"Redis server not available: {e}. Caching disabled.")
        class DummyRedis:
            def get(self, key):
                return None
            def set(self, key, value, ex=None):
                pass
        redis_client = DummyRedis()
except ImportError as e:
    logger.warning(f"redis-py not installed: {e}. Caching disabled.")
    class DummyRedis:
        def get(self, key):
            return None
        def set(self, key, value, ex=None):
            pass
    redis_client = DummyRedis()

app = FastAPI(
    title="Philomena Cunk API",
    description="An API for asking philosophical, sarcastic, and historically informed questions to an AI inspired by Philomena Cunk.",
    version="1.0.0"
)

# CORS setup
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your CSV as a knowledge base

# Use a short temporary text as the knowledge base
temp_text = (
    "Philosophy is the study of general and fundamental questions, such as those "
    "about existence, reason, knowledge, values, mind, and language. "
    "Socrates was a classical Greek philosopher credited as one of the founders of "
    "Western philosophy. Philomena Cunk is a fictional character known for her "
    "satirical takes on history and philosophy."
)

# Write the temp_text to a temporary file for loading
with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
    f.write(temp_text)
    temp_text_path = f.name

loader = TextLoader(file_path=temp_text_path)
documents = loader.load()
docs = documents


# Choose between OpenAI and HuggingFace based on environment variable
USE_HF = os.getenv("USE_HF", "false").lower() == "true"

if USE_HF:
    # transformers must be installed for HuggingFacePipeline
    from transformers import pipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # You can change the model name as needed
    hf_model_name = os.getenv("HF_MODEL_NAME", "gpt2")
    hf_pipe = pipeline("text-generation", model=hf_model_name)
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    # Use an open-source HuggingFace embedding model
    hf_embedding_model = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = HuggingFaceEmbeddings(model_name=hf_embedding_model)
else:
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)  # Use ChatOpenAI for chat models
    embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
)

class ChatRequest(BaseModel):
    question: str = "what is the meaning of life?"


# Wikipedia tool setup

# Wrappers to add logging when tools are used
def wikipedia_logged(*args, **kwargs):
    logger.info(f"[TOOL] Wikipedia tool used. Args: {args}, Kwargs: {kwargs}")
    try:
        result = wikipedia.run(*args, **kwargs)
        logger.info(f"[TOOL] Wikipedia result: {result}")
        return result
    except Exception as e:
        logger.error(f"[TOOL] Wikipedia tool error: {e}")
        raise

def duckduckgo_logged(*args, **kwargs):
    logger.info(f"[TOOL] DuckDuckGo tool used. Args: {args}, Kwargs: {kwargs}")
    try:
        result = duckduckgo_search.run(*args, **kwargs)
        logger.info(f"[TOOL] DuckDuckGo result: {result}")
        return result
    except Exception as e:
        logger.error(f"[TOOL] DuckDuckGo tool error: {e}")
        raise

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
duckduckgo_search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Wikipedia",
        description="Useful for answering factual or historical questions.",
        func=wikipedia_logged,
    ),
    Tool(
        name="DuckDuckGo_Search",
        description="Useful for searching the web for current events or general information.",
        func=duckduckgo_logged,
    ),
]

# LangGraph ReAct agent setup
react_agent = create_react_agent(llm, tools)




@app.post(
    "/ask", summary="Ask the Philomena Cunk", tags=["Philosophy QA"]
)
async def ask_philosopher(request: ChatRequest):
    question = request.question.strip().lower()
    cache_key = f"cunk:qa:{question}"
    # Check Redis cache first
    cached_answer = redis_client.get(cache_key)
    if cached_answer:
        logger.info(f"[REDIS CACHE] Returning cached answer for: {question}")
        return {"answer": cached_answer}

    prompt = (
        "You are Philomena Cunk, a satirical British presenter. "
        "Respond to the following question with wit, sarcasm, and real historical or philosophical context. "
        "If you use Wikipedia, add your own humorous misunderstanding. "
        "Keep your answer under 100 words. Always include a witty remark. End with a rhetorical question if possible. "
        "Examples:\n"
        "Q: What is the meaning of life?\n"
        "A: Well, some say it's 42, but I think it's mostly about finding the remote control before your tea goes cold.\n"
        "Q: Who was Socrates?\n"
        "A: Socrates was a Greek philosopher who asked so many questions, people eventually made him drink poison just to get some peace and quiet.\n"
        f"Q: {request.question}\nA:"
    )
    # Use LangGraph ReAct agent to get the answer
    result = react_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])
    answer = ""
    if messages and hasattr(messages[-1], "content"):
        answer = messages[-1].content
    else:
        answer = str(result)
    # Store in Redis cache (optionally set TTL, e.g., 1 day)
    redis_client.set(cache_key, answer, ex=86400)
    return {"answer": answer}
