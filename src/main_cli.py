import os
import sys
import logging
from dotenv import load_dotenv

# LlamaIndex Core Imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

# LlamaIndex Integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Project Imports
from src.config import EMBEDDING_MODEL_NAME, CACHE_DIR, INDEX_CACHE_DIR
from src.llm_client import OpenRouterLLM
from src.pdf_processor import get_pdf_id, extract_text_from_pdf, chunk_text

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_llama_index_settings():
    """
    Configures and initializes LlamaIndex settings for the LLM and embedding model.
    """
    try:
        logging.info("--- Configuring LlamaIndex Settings ---")

        # Load API keys and other environment variables
        load_dotenv()
        if not os.getenv("OPENROUTER_API_KEY"):
            logging.error("FATAL: OPENROUTER_API_KEY not found in environment variables. Please create a .env file.")
            sys.exit(1)

        # Set up a callback manager for debugging if needed
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        # Settings.callback_manager = callback_manager # Uncomment for detailed debug tracing

        # Configure the embedding model
        logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=f"{CACHE_DIR}/models" # Cache sentence-transformer models
        )
        logging.info("Embedding model initialized.")

        # Configure the custom LLM
        logging.info("Configuring custom LLM for OpenRouter...")
        Settings.llm = OpenRouterLLM()
        logging.info(f"Custom LLM configured with model: {Settings.llm.model_name}")

        return True
    except Exception as e:
        logging.error(f"Failed to configure LlamaIndex settings: {e}")
        return False

def get_or_build_index(pdf_path: str):
    """
    Checks for a cached index for the given PDF. If not found, it builds,
    saves, and returns a new index.
    """
    pdf_id = get_pdf_id(pdf_path)
    if not pdf_id:
        return None # Error occurred in hashing

    index_path = os.path.join(INDEX_CACHE_DIR, pdf_id)

    # --- Load from Cache if available ---
    if os.path.exists(index_path):
        try:
            logging.info(f"Loading cached index for '{os.path.basename(pdf_path)}' from: {index_path}")
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            index = load_index_from_storage(storage_context)
            logging.info("Successfully loaded index from cache.")
            return index
        except Exception as e:
            logging.warning(f"Failed to load cached index: {e}. Re-building from scratch.")

    # --- Build from Scratch if not in cache ---
    logging.info("No valid cache found. Building new index...")

    # 1. Extract Text
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text:
        logging.error("Failed to extract text. Cannot build index.")
        return None

    # 2. Chunk Text into Nodes
    nodes = chunk_text(extracted_text)
    if not nodes:
        logging.error("Failed to generate text chunks (nodes). Cannot build index.")
        return None

    # 3. Build and Persist Vector Index
    try:
        logging.info("Building vector index. This may take a moment...")
        # Enable progress bar for user feedback during indexing
        index = VectorStoreIndex(nodes, show_progress=True)

        logging.info(f"Persisting index to cache for future use at: {index_path}")
        os.makedirs(index_path, exist_ok=True)
        index.storage_context.persist(persist_dir=index_path)
        logging.info("Successfully built and cached new index.")
        return index
    except Exception as e:
        logging.error(f"Error building vector index: {e}")
        return None

def query_index_interactive(index):
    """
    Handles the interactive user query loop.
    """
    if not index:
        logging.error("Index not available for querying.")
        return

    try:
        # Create a query engine with a streaming response mode for better UX
        query_engine = index.as_query_engine(streaming=False) # Set to False if streaming is not supported/needed
    except Exception as e:
        logging.error(f"Error creating query engine: {e}")
        return

    print("\n--- Ready to Answer Questions ---")
    print("Enter your question about the PDF. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_query = input("\nQuery: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['quit', 'exit']:
                print("Exiting query mode.")
                break

            print("\n--- Answer ---")
            response_object = query_engine.query(user_query)

            if hasattr(response_object, 'response_gen'): # Streaming response
                for chunk in response_object.response_gen:
                    print(chunk, end="", flush=True)
                print()
            else: # Regular response
                print(response_object.response)

            # Optional: Show source nodes
            show_sources = input("\nShow source text chunks? (yes/no, default: no): ").strip().lower()
            if show_sources == 'yes' and response_object.source_nodes:
                print("\n--- Source Nodes Used ---")
                for i, source_node in enumerate(response_object.source_nodes):
                    print(f"Source Node {i+1} (Score: {source_node.score:.4f}):")
                    print(f"'{source_node.get_content()[:350].strip()}...'")
                print("-----------------------")
            elif show_sources == 'yes':
                print("No source node information was returned for this response.")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting query mode.")
            break
        except Exception as e:
            logging.error(f"An error occurred during querying: {e}")
            print("Please try a different query or type 'quit' to exit.")

def run():
    """
    The main execution function for the CLI application.
    """
    print("--- Context-Aware RAG Chatbot Initializing ---")

    # 1. Configure LlamaIndex and check for API keys
    if not configure_llama_index_settings():
        sys.exit(1)

    # 2. Get a valid PDF path from the user
    while True:
        pdf_file_path = input("Enter the full path to your PDF file: ").strip()
        if os.path.exists(pdf_file_path) and pdf_file_path.lower().endswith(".pdf"):
            break
        elif not os.path.exists(pdf_file_path):
            logging.error(f"File not found at '{pdf_file_path}'. Please check the path.")
        else:
            logging.error(f"The file '{os.path.basename(pdf_file_path)}' is not a valid PDF file.")

    # 3. Get or Build the Vector Index
    index = get_or_build_index(pdf_file_path)
    if not index:
        logging.error("Failed to get or build the vector index. Cannot proceed.")
        sys.exit(1)

    # 4. Start the Interactive Query Loop
    query_index_interactive(index)

    print("\n--- Script Finished ---")