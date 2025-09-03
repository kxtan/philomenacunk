import logging
from langchain.agents import Tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from backend.services.llm_service import llm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("philomena_cunk")

# Wikipedia tool setup & wrappers to add logging when tools are used
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
duckduckgo_search = DuckDuckGoSearchRun()


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


# Refiner agent for output control
def refine_answer(question, answer):
    """
    Uses the LLM to rewrite the answer to be concise, witty, and in the Philomena Cunk style.
    """
    refine_prompt = (
        "You are a critic and editor for Philomena Cunk. "
        "Given the following question and answer, rewrite the answer to be even more concise, witty, and in the Philomena Cunk style. "
        "Keep it under 100 words, ensure it is on-topic, and always end with a rhetorical question or a humorous twist. "
        "If the answer is already good, you may return it unchanged.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Refined Answer:"
    )
    # Use the same LLM for refinement (could be swapped for a different one)
    result = llm.invoke(refine_prompt)
    # Handle both string and object outputs
    if hasattr(result, "content"):
        return result.content.strip()
    return str(result).strip()