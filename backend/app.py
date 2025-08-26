from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFacePipeline
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI()

# Load your CSV as a knowledge base
csv_path = "data/test_data.csv"  # Change to your CSV file path
loader = CSVLoader(file_path=csv_path)
documents = loader.load()

# Split documents into smaller chunks to avoid token limit errors
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
docs = text_splitter.split_documents(documents)


# Choose between OpenAI and HuggingFace based on environment variable
USE_HF = os.getenv("USE_HF", "false").lower() == "true"

if USE_HF:
    # transformers must be installed for HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModel
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # You can change the model name as needed
    hf_model_name = os.getenv("HF_MODEL_NAME", "gpt2")
    hf_pipe = pipeline("text-generation", model=hf_model_name)
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    # Use an open-source HuggingFace embedding model
    hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=hf_embedding_model)
else:
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
    question: str = "what is the meaning of life?"



@app.post("/chat")
async def chat(request: ChatRequest):
    # Use a prompt template for sarcasm, philosophy, and history
    prompt_template = (
        "Respond to the following question with a sarcastic tone, "
        "infused with philosophical and historical context. "
        "Reference philosophers or historical events as appropriate, "
        "and ensure the answer is witty and dry.\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    # Format the prompt for the LLM
    prompt = prompt_template.format(question=request.question)
    # Get the answer from the LLM using the prompt
    answer = qa_chain.run(prompt)
    # Return only the output text
    return {"answer": answer}
