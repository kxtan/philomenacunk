from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

# Import modularized components
from backend.routes.chat_routes import router as chat_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("philomena_cunk")
load_dotenv()

app = FastAPI(
    title="Philomena Cunk API",
    description=(
        "An API for asking philosophical, sarcastic, and historically "
        "informed questions to an AI inspired by Philomena Cunk."
    ),
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

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Philomena Cunk API is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
