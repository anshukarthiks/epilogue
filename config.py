import os

from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class Settings:
    OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = get_setting("OLLAMA_MODEL", "qwen3:8b")
    OLLAMA_EMBEDDING_MODEL = get_setting("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_NUM_GPU = int(os.environ["OLLAMA_NUM_GPU"]) if os.getenv("OLLAMA_NUM_GPU") else None
    OLLAMA_NUM_CTX = int(get_setting("OLLAMA_NUM_CTX", "4096"))
    OLLAMA_NUM_PREDICT = int(get_setting("OLLAMA_NUM_PREDICT", "512"))
    CHROMA_PERSIST_DIRECTORY = get_setting("CHROMA_PERSIST_DIRECTORY", os.path.join(os.getcwd(), "data", "chroma"))
    CHROMA_COLLECTION_NAME = get_setting("CHROMA_COLLECTION_NAME", "epilogue_documents")
    CHUNK_SIZE = int(get_setting("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(get_setting("CHUNK_OVERLAP", "150"))
    RETRIEVAL_TOP_K = int(get_setting("RETRIEVAL_TOP_K", "4"))
    MAX_RETRIES = int(get_setting("MAX_RETRIES", "2"))
    MAX_UPLOAD_SIZE = int(get_setting("MAX_UPLOAD_SIZE", str(16 * 1024 * 1024)))
    UPLOAD_DIRECTORY = get_setting("UPLOAD_DIRECTORY", os.path.join(os.getcwd(), "data", "uploads"))


settings = Settings()
