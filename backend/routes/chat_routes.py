from fastapi import APIRouter
from backend.models.chat_models import ChatRequest
from backend.services.redis_service import redis_client
from backend.services.agent_service import react_agent, refine_answer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("philomena_cunk")

router = APIRouter()


@router.post("/ask", summary="Ask the Philomena Cunk", tags=["Philosophy QA"])
async def ask_philosopher(request: ChatRequest):
    question = request.question.strip().lower()
    cache_key = f"cunk:qa:{question}"
    # Check Redis cache first
    cached_answer = redis_client.get(cache_key)
    if cached_answer:
        logger.info(f"[REDIS CACHE] Returning cached answer for: {question}")
        return {"answer": cached_answer}
    # Build conversation context from history (last 5 exchanges)
    history = request.history if request.history else []
    history_prompt = ""
    for turn in history[-5:]:
        q = turn.get("question", "")
        a = turn.get("answer", "")
        if q and a:
            history_prompt += f"Q: {q}\nA: {a}\n"
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
        f"{history_prompt}Q: {request.question}\nA:"
    )
    # Use LangGraph ReAct agent to get the answer
    result = react_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])
    answer = ""
    if messages and hasattr(messages[-1], "content"):
        answer = messages[-1].content
    else:
        answer = str(result)
    # Refine the answer using the refiner agent
    refined_answer = refine_answer(request.question, answer)
    # Store in Redis cache (optionally set TTL, e.g., 1 day)
    redis_client.set(cache_key, refined_answer, ex=86400)
    return {"answer": refined_answer}