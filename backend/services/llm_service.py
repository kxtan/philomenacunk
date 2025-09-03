import os
import logging
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("philomena_cunk")
load_dotenv()

# Load your CSV as a knowledge base
temp_text = (
    "Philosophy is the study of general and fundamental questions, such as those "
    "about existence, reason, knowledge, values, mind, and language. "
    "Socrates was a classical Greek philosopher credited as one of the founders of "
    "Western philosophy. Philomena Cunk is a fictional character known for her "
    "satirical takes on history and philosophy."
)
with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
    f.write(temp_text)
    temp_text_path = f.name
loader = TextLoader(file_path=temp_text_path)
documents = loader.load()
docs = documents

# Choose between OpenAI and HuggingFace based on environment variable
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
if USE_OPENROUTER:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model_name = os.getenv("OPENROUTER_MODEL_NAME")
    llm = ChatOpenAI(
        model=openrouter_model_name,
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=openrouter_api_key,
    )
    hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=hf_embedding_model)
else:
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=hf_embedding_model)

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
)