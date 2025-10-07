# --- Configuration ---
# This file centralizes all key parameters for the application.
# Modify these values to tune the behavior of the RAG pipeline.

# --- Embedding Model ---
# The model used to convert text chunks into numerical vectors (embeddings).
# 'all-MiniLM-L6-v2' is a good default: fast, efficient, and high-quality.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM Model ---
# The large language model used to generate answers.
# Sourced from OpenRouter. See their documentation for available models.
# 'qwen/qwen3-30b-a3b:free' is a capable free model.
LLM_MODEL_NAME = "qwen/qwen3-30b-a3b:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# --- LLM Parameters ---
# CONTEXT_WINDOW: The maximum number of tokens the LLM can consider.
# MAX_TOKENS_OUTPUT: The maximum number of tokens for the generated answer.
# TEMPERATURE: Controls the "creativity" of the LLM. Lower is more deterministic.
# REQUEST_TIMEOUT: How long to wait for a response from the API.
CONTEXT_WINDOW = 41984
MAX_TOKENS_OUTPUT = 4096
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 120

# --- Text Chunking ---
# CHUNK_SIZE: The target size of each text chunk in tokens.
# CHUNK_OVERLAP: The number of tokens to overlap between adjacent chunks.
# This helps maintain context across chunk boundaries.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Caching ---
# The directory where sentence-transformer models and generated indexes are stored.
# This is ignored by Git to avoid committing large files.
# SECURITY NOTE: The cache contains processed text from your PDFs.
# If you handle sensitive documents, ensure this directory is properly secured.
CACHE_DIR = ".cache"
INDEX_CACHE_DIR = f"{CACHE_DIR}/indexes"

# --- Optional API Metadata ---
# These are sent as headers to OpenRouter and can be useful for tracking/analytics.
# Replace with your own project details if desired.
HTTP_REFERER = "https://github.com/your-repo/your-project"
X_TITLE = "Context-Aware RAG Chatbot"