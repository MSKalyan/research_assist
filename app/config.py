import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COLLECTION = "research_documents"

PASSING_SCORE = int(os.getenv("PASSING_SCORE", "7"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MAX_EXPANDED_QUERIES = int(os.getenv("MAX_EXPANDED_QUERIES", "5"))
MAX_TOP_RESULTS = int(os.getenv("MAX_TOP_RESULTS", "10"))

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
DEFAULT_EMBEDDING = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment.")