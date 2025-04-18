"""
Configuration module for the BCG Multi-Agent & Multimodal AI Platform.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store configuration
VECTOR_STORE_PATH = EMBEDDINGS_DIR / "chroma"

# Document processing
USE_OCR = True
EXTRACT_CHARTS = True

# Ethical AI configuration
ENABLE_BIAS_DETECTION = True
ENABLE_CONTENT_FILTERING = True
ENABLE_EXPLANATIONS = True

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"