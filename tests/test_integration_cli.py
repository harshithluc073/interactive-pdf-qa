import os
import pytest
import logging
import io  # Import the io module
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from src.main_cli import run

# --- Fixtures ---

@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    """
    Mocks environment variables, stdin for user input, and cache directories.
    """
    # 1. Mock the API key
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-api-key")

    # 2. Isolate cache to a temporary directory
    temp_cache_dir = tmp_path / ".cache"
    temp_index_dir = temp_cache_dir / "indexes"
    monkeypatch.setattr("src.config.CACHE_DIR", str(temp_cache_dir))
    monkeypatch.setattr("src.config.INDEX_CACHE_DIR", str(temp_index_dir))

    # 3. Mock user inputs by patching sys.stdin
    pdf_path = os.path.join(os.path.dirname(__file__), "test_document.pdf")
    input_text = "\n".join([
        pdf_path,
        "What is AI?",
        "no",
        "quit"
    ])
    monkeypatch.setattr('sys.stdin', io.StringIO(input_text))

@pytest.fixture
def mock_llm(monkeypatch):
    """Mocks the OpenRouterLLM to prevent real API calls."""
    mock_response_content = "AI is the simulation of human intelligence in machines."
    mock_chat_response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content=mock_response_content),
        raw={"id": "mock_response"}
    )
    mock_chat = MagicMock(return_value=mock_chat_response)
    monkeypatch.setattr("src.llm_client.OpenRouterLLM.chat", mock_chat)
    return mock_chat

# --- Integration Test ---

def test_cli_end_to_end_flow(capsys, caplog, mock_env, mock_llm):
    """
    Tests the full CLI flow, separating assertions for stdout and logs.
    """
    with caplog.at_level(logging.INFO):
        run()

    # --- Assert ---
    captured = capsys.readouterr()
    stdout = captured.out
    log_text = caplog.text

    # 1. Assert against console output (user-facing prompts and responses)
    assert "Context-Aware RAG Chatbot Initializing" in stdout
    assert "Enter the full path to your PDF file:" in stdout
    assert "Ready to Answer Questions" in stdout
    assert "Query:" in stdout
    assert "--- Answer ---" in stdout
    assert "AI is the simulation of human intelligence in machines." in stdout
    assert "Exiting query mode." in stdout
    assert "Script Finished" in stdout

    # 2. Assert against log output (internal state)
    assert "Configuring LlamaIndex Settings" in log_text
    assert "Building new index..." in log_text
    assert "Successfully built and cached new index" in log_text

    # 3. Verify the LLM was called correctly
    llm_call_args = mock_llm.call_args
    assert llm_call_args is not None
    messages = llm_call_args[0][0]
    user_message_content = messages[-1].content
    assert "What is AI?" in user_message_content