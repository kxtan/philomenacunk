from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFacePipeline
import tempfile
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()


app = FastAPI(
    title="Philosopher Cunk API",
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
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)  # Use ChatOpenAI for chat models
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
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for answering factual or historical questions."
    )
]

# Initialize agent for autonomous tool use
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)


@app.post(
    "/ask", summary="Ask the Philosomena Cunk", tags=["Philosophy QA"]
)
async def ask_philosopher(request: ChatRequest):
    prompt = (
        "Respond as Philomena Cunk: sarcastic, witty, with philosophical and historical context. "
        "Use Wikipedia or other tools if needed. "
        f"Question: {request.question}\nAnswer:"
    )
    answer = agent.run(prompt)
    return {"answer": answer}
