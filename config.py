import os
from dataclasses import dataclass
from typing import List

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_chunks_per_doc: int = 100

@dataclass
class VectorStoreConfig:
    """Configuration for ChromaDB vector store"""
    collection_name: str = "knowledge_base"
    distance_metric: str = "cosine"
    n_results: int = 3
    confidence_threshold: float = 0.6

@dataclass
class WebSearchConfig:
    """Configuration for Tavily web search"""
    api_key: str = os.getenv("TAVILY_API_KEY", "")
    search_depth: str = "advanced"
    max_results: int = 3
    timeout: int = 10

@dataclass
class LLMConfig:
    """Configuration for OpenAI LLM"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 600
    temperature: float = 0.7
    factual_temperature: float = 0.3

@dataclass
class ServerConfig:
    """Configuration for backend server"""
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_url: str = "http://localhost:8501"
    log_level: str = "info"

@dataclass
class AppConfig:
    """Main application configuration"""
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    web_search: WebSearchConfig = WebSearchConfig()
    llm: LLMConfig = LLMConfig()
    server: ServerConfig = ServerConfig()
    
    # File processing
    supported_file_types: List[str] = None
    max_file_size_mb: int = 10
    
    def __post_init__(self):
        if self.supported_file_types is None:
            self.supported_file_types = ['.txt', '.pdf', '.docx']

# Global config instance
config = AppConfig()

# Environment validation
def validate_environment():
    """Validate required environment variables"""
    missing_vars = []
    
    if not config.llm.api_key:
        missing_vars.append("OPENAI_API_KEY")
    
    if not config.web_search.api_key:
        missing_vars.append("TAVILY_API_KEY (optional but recommended)")
    
    if missing_vars:
        print("Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("The application will work with limited functionality.")
    
    return len([var for var in missing_vars if "optional" not in var]) == 0

# Query analysis keywords
QUERY_KEYWORDS = {
    "time_sensitive": ["current", "latest", "recent", "today", "now", "2024", "2025", "this week", "this month"],
    "factual": ["what is", "who is", "when did", "where is", "how many", "define", "definition"],
    "analytical": ["analyze", "compare", "explain why", "what are the implications", "how does", "why do"],
    "web_priority": ["news", "breaking", "update", "announcement", "released"]
}