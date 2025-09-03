from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatRequest(BaseModel):
    question: str = "what is the meaning of life?"
    history: Optional[List[Dict[str, str]]] = []