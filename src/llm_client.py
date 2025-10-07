import os
import sys
import requests
import json
from typing import Any, Sequence, Optional

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

from src.config import (
    LLM_MODEL_NAME,
    OPENROUTER_API_BASE,
    CONTEXT_WINDOW,
    TEMPERATURE,
    MAX_TOKENS_OUTPUT,
    REQUEST_TIMEOUT,
    HTTP_REFERER,
    X_TITLE,
)

class OpenRouterLLM(CustomLLM):
    """
    Custom LlamaIndex LLM class to interact with the OpenRouter API.
    """
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
        """
        Provides metadata about the LLM.
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Handles synchronous chat completions.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

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
                timeout=REQUEST_TIMEOUT,
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
            # More specific error handling could be added here (e.g., for 4xx vs 5xx errors)
            print(f"API Error calling OpenRouter: {e}", file=sys.stderr)
            raise
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing OpenRouter response: {e}. Response: {response_json}", file=sys.stderr)
            raise

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Handles synchronous text completions (not streaming).
        """
        response = self.chat([ChatMessage(role=MessageRole.USER, content=prompt)], **kwargs)
        return CompletionResponse(text=response.message.content, raw=response.raw)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Streaming chat is not implemented.
        """
        raise NotImplementedError("Streaming chat is not implemented for OpenRouterLLM.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Streaming completion is not implemented.
        """
        raise NotImplementedError("Streaming completion is not implemented for OpenRouterLLM.")