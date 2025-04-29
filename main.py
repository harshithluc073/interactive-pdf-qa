import os
import sys
import requests
import json
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from typing import Any, Sequence, Optional

# LlamaIndex Core Imports
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import Document, Node
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

# LlamaIndex Integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Environment Variable Loading ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "qwen/qwen3-30b-a3b:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
CONTEXT_WINDOW = 41984
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TEMPERATURE = 0.1
MAX_TOKENS_OUTPUT = 4096
REQUEST_TIMEOUT = 120
HTTP_REFERER = None
X_TITLE = None

# --- Custom LLM Class (OpenRouterLLM - unchanged from previous version) ---
class OpenRouterLLM(CustomLLM):
    model_name: str = LLM_MODEL_NAME
    api_key: str = ""
    api_base: str = OPENROUTER_API_BASE
    context_window: int = CONTEXT_WINDOW
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS_OUTPUT
    http_referer: Optional[str] = HTTP_REFERER
    x_title: Optional[str] = X_TITLE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
             raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer: headers["HTTP-Referer"] = self.http_referer
        if self.x_title: headers["X-Title"] = self.x_title
        api_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        data = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        try:
            response = requests.post(
                url=f"{self.api_base}/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            response_json = response.json()
            if not response_json.get("choices") or not response_json["choices"][0].get("message"):
                 raise ValueError(f"Unexpected response format from OpenRouter: {response_json}")
            content = response_json["choices"][0]["message"]["content"]
            role_str = response_json["choices"][0]["message"].get("role", "assistant")
            return ChatResponse(
                message=ChatMessage(role=MessageRole(role_str), content=content),
                raw=response_json,
            )
        except requests.exceptions.RequestException as e:
            print(f"API Error calling OpenRouter: {e}", file=sys.stderr)
            raise
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing OpenRouter response: {e}. Response: {response_json}", file=sys.stderr)
            raise

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.chat([ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs)
        return CompletionResponse(text=response.message.content, raw=response.raw)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("Streaming chat is not implemented for OpenRouterLLM.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("Streaming completion is not implemented for OpenRouterLLM.")

# --- Helper Functions ---
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'", file=sys.stderr)
        return None
    try:
        print(f"Attempting to parse: {pdf_path}")
        reader = PdfReader(pdf_path)
        if not reader.pages:
             print(f"Warning: PDF file '{os.path.basename(pdf_path)}' contains no pages.", file=sys.stderr)
             return None
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            # else:
            #     print(f"Debug: No text extracted from page {i+1}", file=sys.stderr)

        if not text.strip():
             print(f"Warning: No text content could be extracted from '{os.path.basename(pdf_path)}'. It might be image-based or empty.", file=sys.stderr)
             return None
        print(f"Successfully extracted text from {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        print(f"Error reading PDF '{pdf_path}': {e}", file=sys.stderr)
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text: return []
    try:
        document = Document(text=text)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents([document])
        print(f"Split text into {len(nodes)} nodes (chunks).")
        if not nodes:
             print("Warning: Text splitting resulted in zero nodes.", file=sys.stderr)
        return nodes
    except Exception as e:
        print(f"Error during text chunking: {e}", file=sys.stderr)
        return []

# --- Core Logic Functions ---
def configure_llama_index_settings():
    try:
        print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model initialized.")
        print("Configuring custom LLM for OpenRouter...")
        Settings.llm = OpenRouterLLM()
        print(f"Custom LLM configured with model: {Settings.llm.model_name}")
        return True
    except ValueError as ve:
         print(f"Configuration Error: {ve}", file=sys.stderr)
         return False
    except Exception as e:
        print(f"Error configuring LlamaIndex settings: {e}", file=sys.stderr)
        return False

def build_vector_index(nodes):
    if not nodes: return None
    try:
        print("\n--- Building Vector Index ---")
        # Disable showing progress bars during indexing for cleaner interactive console
        index = VectorStoreIndex(nodes, show_progress=False)
        print(f"Successfully built vector index.")
        print("---------------------------")
        return index
    except Exception as e:
        print(f"Error building vector index: {e}", file=sys.stderr)
        return None

def query_index_interactive(index):
    """Handles the interactive querying loop."""
    if not index:
        print("Error: Index not available for querying.", file=sys.stderr)
        return

    # Create query engine once
    try:
        query_engine = index.as_query_engine()
    except Exception as e:
        print(f"Error creating query engine: {e}", file=sys.stderr)
        return

    print("\n--- Ready to Answer Questions ---")
    print("Enter your question about the PDF. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_query = input("\nQuery: ").strip()
            if not user_query:
                continue # Ask again if empty input
            if user_query.lower() in ['quit', 'exit']:
                print("Exiting query mode.")
                break

            print("Generating response...")
            response_object = query_engine.query(user_query)

            if response_object:
                print("\n--- Answer ---")
                print(response_object.response)

                # Optional: Show source nodes
                show_sources = input("Show source text chunks? (yes/no, default: no): ").strip().lower()
                if show_sources == 'yes' and response_object.source_nodes:
                    print("\n--- Source Nodes Used ---")
                    for i, source_node in enumerate(response_object.source_nodes):
                        print(f"\nSource Node {i+1} (Score: {source_node.score:.4f}):")
                        print(source_node.get_content()[:300].strip() + "...") # Show a bit more context
                    print("-----------------------")
                elif show_sources == 'yes':
                    print("No source node information available for this response.")

            else:
                # This might happen if the LLM call fails internally in query_engine
                print("Sorry, I could not generate a response for that query.", file=sys.stderr)

        except EOFError: # Handle Ctrl+D
             print("\nExiting query mode.")
             break
        except KeyboardInterrupt: # Handle Ctrl+C
             print("\nExiting query mode.")
             break
        except Exception as e:
             # Catch errors during a specific query, allow user to continue
             print(f"\nAn error occurred during querying: {e}", file=sys.stderr)
             print("Please try a different query or type 'quit' to exit.")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- PDF RAG Chatbot Initializing ---")

    # 1. Configure LlamaIndex Settings
    if not configure_llama_index_settings():
        sys.exit(1)

    # 2. Get PDF Path from User
    while True:
        pdf_file_path = input("Enter the full path to your PDF file: ").strip()
        if os.path.exists(pdf_file_path) and pdf_file_path.lower().endswith(".pdf"):
            break
        elif os.path.exists(pdf_file_path):
             print(f"Error: The file '{pdf_file_path}' is not a PDF file.", file=sys.stderr)
        else:
             print(f"Error: File not found at '{pdf_file_path}'. Please check the path and try again.", file=sys.stderr)

    # 3. Extract Text
    extracted_text = extract_text_from_pdf(pdf_file_path)
    if not extracted_text:
        print("Failed to extract text, cannot proceed.", file=sys.stderr)
        sys.exit(1)
    print(f"\nTotal characters extracted: {len(extracted_text)}")

    # 4. Chunk Text into Nodes
    nodes = chunk_text(extracted_text)
    if not nodes:
        print("Failed to generate nodes (chunks), cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # 5. Build Vector Index
    index = build_vector_index(nodes)
    if not index:
        print("Failed to build vector index, cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # 6. Enter Interactive Query Loop
    query_index_interactive(index)

    print("\n--- Script Finished ---")