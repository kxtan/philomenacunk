from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI()

# Load your CSV as a knowledge base
csv_path = "data/test_data.csv"  # Change to your CSV file path
loader = CSVLoader(file_path=csv_path)
documents = loader.load()

# Split documents into smaller chunks to avoid token limit errors
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Set up the retriever and LLM
llm = OpenAI(temperature=0)  # Requires OPENAI_API_KEY env variable
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    answer = qa_chain.run(request.question)
    return {"answer": answer}