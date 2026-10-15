"""Local runtime configuration. No cloud credentials or services are used."""

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Set before Chroma or LangChain are imported.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_upload_bytes: int = 25 * 1024 * 1024

    def __post_init__(self):
        parsed = urlparse(self.ollama_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "localhost", "127.0.0.1", "::1", "ollama", "host.docker.internal"
        } or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("OLLAMA_BASE_URL must point to the local Ollama service over HTTP.")
        for model in (self.chat_model, self.embedding_model):
            if not model.strip() or "cloud" in model.lower() or "/" in model:
                raise ValueError("Choose a local Ollama model, such as llama3.1:8b; cloud models are disabled.")
        if self.chunk_size < 100 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than the chunk size (at least 100).")

    @property
    def db_dir(self) -> Path:
        return self.data_dir.resolve() / "chroma_db"

    @property
    def collection_name(self) -> str:
        # Different vector models or chunking configurations must never mix.
        key = f"{self.embedding_model}|{self.chunk_size}|{self.chunk_overlap}|v1"
        return "documents_" + sha256(key.encode()).hexdigest()[:16]

    @classmethod
    def from_env(cls):
        load_dotenv(Path(__file__).parent / ".env", override=False)
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", str(Path(__file__).parent / "data"))),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            chat_model=os.getenv("LLM_MODEL", "llama3.1:8b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        )
