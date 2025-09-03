import os
import logging
from dotenv import load_dotenv

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

    class DummyRedisFallback:
        def get(self, key):
            return None

        def set(self, key, value, ex=None):
            pass

    redis_client = DummyRedisFallback()