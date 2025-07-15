import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load environment variables
load_dotenv()

# ----------------------------
# Load paths from .env
# ----------------------------
GGUF_MODEL_PATH = os.getenv("LOCAL_MODAL_PATH")  # Path to GGUF model
EMBED_MODEL_PATH = os.getenv("LOCAL_EMBED_MODEL_PATH")  # Path to MiniLM model

# ----------------------------
# Cache the models on first use
# ----------------------------
_llm_instance = None
_embedder_instance = None


def load_llm(n_ctx: int = 2048, n_threads: int = 6):
    """Load GGUF LLM model using llama-cpp-python"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=n_ctx,
            n_threads=n_threads,
            f16_kv=True,
            use_mlock=True,
            verbose=False
        )
    return _llm_instance


def load_embedder():
    """Load local MiniLM embedding model"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SentenceTransformer(EMBED_MODEL_PATH)
    return _embedder_instance
