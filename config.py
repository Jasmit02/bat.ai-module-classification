"""
Configuration settings for the Module Classification System.
"""
from dataclasses import dataclass
from typing import Dict
import os

@dataclass
class Config:
    """Configuration settings for the Module Classification System."""
    # API Keys
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "default_key")
    
    # Model Configurations
    chat_models: Dict[str, str] = None
    default_chat_model: str = "nvdev/meta/llama-3.1-405b-instruct"
    embed_model: str = "nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2"
    rerank_model: str = "nvdev/nvidia/llama-3.2-nv-rerankqa-1b-v2"
    
    # Search Parameters
    k_value: int = 50
    score_threshold: float = 0.20
    top_n_rerank: int = 20
    
    # Chunk Sizes
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512
    
    # FAISS Index
    index_path: str = os.getenv("FAISS_INDEX_PATH", "faiss-db")
    allow_dangerous_deserialization: bool = True
    
       
    
    # MultiQuery Configuration
    use_multi_query: bool = True
    num_alt_queries: int = 5
    show_generated_queries: bool = True
    
        
    # Evaluation Parameters
    sample_size: int = 10
    num_threads: int = 50
    
    def __post_init__(self):
        """Initialize default values that depend on other fields."""
        if self.chat_models is None:
            self.chat_models = {
                "Llama 3.1": "nvdev/meta/llama-3.1-405b-instruct",
                "Deepseek-R1": "deepseek-ai/deepseek-r1-7b-instruct",
                "Mistral-22B": "nvdev/mistralai/mixtral-8x22b-instruct-v0.1"
            }

# Create a default configuration instance
config = Config()