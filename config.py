#!/usr/bin/env python3
"""
Configuration file for RAG-Based Semantic Quote Retrieval System
Contains all configurable parameters and settings
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for model settings"""
    base_model_name: str = "all-MiniLM-L6-v2"
    fine_tuned_model_path: str = "./fine_tuned_quote_model"
    max_sequence_length: int = 512
    embedding_dimension: int = 384
    batch_size: int = 16
    training_epochs: int = 2
    warmup_steps: int = 100
    learning_rate: float = 2e-5

@dataclass
class DataConfig:
    """Configuration for data processing"""
    dataset_name: str = "Abirate/english_quotes"
    processed_data_file: str = "processed_quotes.jsonl"
    embeddings_file: str = "quote_embeddings.pkl"
    faiss_index_file: str = "quotes_faiss_index.bin"
    min_quote_length: int = 10
    max_quote_length: int = 1000
    
@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    use_openai: bool = False
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.3
    
@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    num_evaluation_queries: int = 50
    evaluation_metrics: List[str] = None
    output_dir: str = "./evaluation_results"
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "faithfulness",
                "answer_relevancy", 
                "context_precision",
                "context_recall",
                "context_relevancy"
            ]

@dataclass
class StreamlitConfig:
    """Configuration for Streamlit app"""
    host: str = "localhost"
    port: int = 8501
    title: str = "Semantic Quote Retrieval System"
    page_icon: str = "📚"
    layout: str = "wide"
    max_upload_size: int = 200  # MB
    
@dataclass
class SystemConfig:
    """Overall system configuration"""
    project_name: str = "RAG Quote Retrieval"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    cache_enabled: bool = True
    gpu_enabled: bool = False
    random_seed: int = 42

class Config:
    """Main configuration class that combines all configs"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.rag = RAGConfig()
        self.evaluation = EvaluationConfig()
        self.streamlit = StreamlitConfig()
        self.system = SystemConfig()
        
        # Override with environment variables if present
        self._load_from_env()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Model config from env
        if os.getenv("BASE_MODEL"):
            self.model.base_model_name = os.getenv("BASE_MODEL")
        if os.getenv("BATCH_SIZE"):
            self.model.batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("TRAINING_EPOCHS"):
            self.model.training_epochs = int(os.getenv("TRAINING_EPOCHS"))
            
        # RAG config from env
        if os.getenv("USE_OPENAI"):
            self.rag.use_openai = os.getenv("USE_OPENAI").lower() == "true"
        if os.getenv("OPENAI_MODEL"):
            self.rag.openai_model = os.getenv("OPENAI_MODEL")
        if os.getenv("TOP_K_DEFAULT"):
            self.rag.top_k_retrieval = int(os.getenv("TOP_K_DEFAULT"))
            
        # Streamlit config from env
        if os.getenv("STREAMLIT_HOST"):
            self.streamlit.host = os.getenv("STREAMLIT_HOST")
        if os.getenv("STREAMLIT_PORT"):
            self.streamlit.port = int(os.getenv("STREAMLIT_PORT"))
            
        # System config from env
        if os.getenv("DEBUG"):
            self.system.debug = os.getenv("DEBUG").lower() == "true"
        if os.getenv("GPU_ENABLED"):
            self.system.gpu_enabled = os.getenv("GPU_ENABLED").lower() == "true"
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            Path(self.model.fine_tuned_model_path).parent,
            Path(self.data.processed_data_file).parent,
            Path(self.evaluation.output_dir),
            Path("logs"),
            Path("temp")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "rag": self.rag.__dict__,
            "evaluation": self.evaluation.__dict__,
            "streamlit": self.streamlit.__dict__,
            "system": self.system.__dict__
        }
    
    def save_to_file(self, filepath: str = "config.json"):
        """Save configuration to JSON file"""
        import json
        
        config_dict = self.to_dict()
        # Remove sensitive information
        if "openai_api_key" in config_dict["rag"]:
            config_dict["rag"]["openai_api_key"] = "***HIDDEN***"
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Configuration saved to {filepath}")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate model config
        if self.model.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.model.training_epochs <= 0:
            errors.append("Training epochs must be positive")
        if not (0 < self.model.learning_rate < 1):
            errors.append("Learning rate must be between 0 and 1")
        
        # Validate RAG config
        if self.rag.use_openai and not self.rag.openai_api_key:
            errors.append("OpenAI API key required when use_openai is True")
        if self.rag.top_k_retrieval <= 0:
            errors.append("Top-k retrieval must be positive")
        if not (0 <= self.rag.temperature <= 2):
            errors.append("Temperature must be between 0 and 2")
        
        # Validate data config
        if self.data.min_quote_length >= self.data.max_quote_length:
            errors.append("Min quote length must be less than max quote length")
        
        # Validate Streamlit config
        if not (1 <= self.streamlit.port <= 65535):
            errors.append("Streamlit port must be between 1 and 65535")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_device(self) -> str:
        """Get the appropriate device (CPU/GPU)"""
        if self.system.gpu_enabled:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"
    
    def setup_logging(self):
        """Setup logging configuration"""
        import logging
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.system.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/rag_system.log'),
                logging.StreamHandler()
            ]
        )
        
        if self.system.debug:
            logging.getLogger().setLevel(logging.DEBUG)

# Global configuration instance
config = Config()

# Configuration presets for different environments
DEVELOPMENT_CONFIG = {
    "system": {"debug": True, "log_level": "DEBUG"},
    "model": {"batch_size": 8, "training_epochs": 1},
    "data": {"min_quote_length": 5},
    "rag": {"top_k_retrieval": 3}
}

PRODUCTION_CONFIG = {
    "system": {"debug": False, "log_level": "INFO", "cache_enabled": True},
    "model": {"batch_size": 32, "training_epochs": 3},
    "rag": {"top_k_retrieval": 10, "similarity_threshold": 0.4}
}

DEMO_CONFIG = {
    "system": {"debug": False, "log_level": "WARNING"},
    "model": {"batch_size": 16, "training_epochs": 1},
    "rag": {"top_k_retrieval": 5, "use_openai": False}
}

def load_preset(preset_name: str):
    """Load a configuration preset"""
    presets = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "demo": DEMO_CONFIG
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = presets[preset_name]
    
    # Apply preset to global config
    for section, settings in preset.items():
        section_obj = getattr(config, section)
        for key, value in settings.items():
            setattr(section_obj, key, value)
    
    print(f"Loaded {preset_name} configuration preset")
    return config

# Helper functions
def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def validate_config() -> bool:
    """Validate the global configuration"""
    return config.validate()

def setup_environment():
    """Setup the complete environment based on configuration"""
    # Setup logging
    config.setup_logging()
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    random.seed(config.system.random_seed)
    np.random.seed(config.system.random_seed)
    
    try:
        import torch
        torch.manual_seed(config.system.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.system.random_seed)
    except ImportError:
        pass
    
    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid configuration")
    
    print(f"Environment setup complete for {config.system.project_name} v{config.system.version}")
    print(f"Device: {config.get_device()}")
    print(f"Debug mode: {config.system.debug}")

if __name__ == "__main__":
    # Demo of configuration usage
    print("RAG Quote Retrieval System - Configuration")
    print("=" * 50)
    
    # Show current configuration
    print("Current Configuration:")
    config.save_to_file("current_config.json")
    
    # Validate configuration
    is_valid = validate_config()
    print(f"Configuration valid: {is_valid}")
    
    # Setup environment
    if is_valid:
        setup_environment()
    
    # Demo preset loading
    print("\nLoading demo preset...")
    demo_config = load_preset("demo")
    print("Demo configuration loaded successfully!")