"""Utility functions for language model and embedding setup.

This module provides factory functions for initializing language models and embeddings
used by the RAG system. It supports multiple model providers (OpenAI, Google GenAI, Ollama)
and can be configured via environment variables.

Example:
    >>> from utils import get_llm, get_embedding_function
    >>> llm = get_llm()  # Get configured language model
    >>> embeddings = get_embedding_function()  # Get embedding model
"""

import ast
import os
import json
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load .env file into os.environ (no-op if file doesn't exist)
load_dotenv()

if os.environ.get("LANGSMITH_TRACING", "false").lower() == "true":
    print("\n** Langsmith Tracing Enabled **\n")



def get_llm() -> BaseChatModel:
    """Initialize and return a language model for chat interactions.

    This function sets up a language model based on available credentials and
    configuration. It supports multiple providers:

    - OpenAI GPT models
    - Google Gemini models
    - Local Ollama models

    The choice of model can be configured through environment variables or
    by modifying the implementation directly.

    Configuration Options:
        - OpenAI:
          ```
          # Required: pip install langchain-openai
          from langchain_openai import ChatOpenAI
          llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
          ```

        - Google GenAI:
          ```
          # Required: pip install langchain-google-genai
          from langchain_google_genai import ChatGoogleGenerativeAI
          llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
          ```

    Returns:
        BaseChatModel: Initialized language model instance

    Raises:
        ValueError: If required environment variables are missing
    """
    if os.environ.get("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            base_url=os.environ.get("OPEN_API_URL"),
        )
        print("\n** Initialized OpenAI LLM **\n")
        return llm

    if os.environ.get("GOOGLE_API_KEY"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0
        )
        print("\n** Initialized Google GenAI LLM **\n")
        return llm

    # Local fallback — requires Ollama running
    llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0.0)
    print("\n** Initialized Ollama LLM **\n")
    return llm


class CustomChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI class to override the invoke method for tracing."""

    def invoke(self, messages, **kwargs):
        """Override invoke to add custom tracing logic."""
        return super().invoke(messages, **kwargs)

def get_custom_llm() -> BaseChatModel:
    if os.environ.get("OPENAI_API_KEY"):
        llm = CustomChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            base_url=os.environ.get("OPEN_API_URL"),
        )
        print("\n** Initialized CustomChatOpenAI LLM **\n")
        return llm

    if os.environ.get("GOOGLE_API_KEY"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0
        )
        print("\n** Initialized Google GenAI LLM **\n")
        return llm

    raise ValueError(
        "No LLM API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY in your environment."
    )


def summarize_conversation(messages: list) -> str:
    """Summarizes a conversation using a language model.

    Args:
        messages (list): A list of messages in the conversation.

    Returns:
        str: A summary of the conversation.
    """
    llm = get_custom_llm()

    # Construct a prompt for summarization
    prompt = f"Please summarize the following conversation and provide a json format of summary, customer_info (strictly json with case sensitive keys - ['name', 'company', 'domain', 'email', 'topic']), specialist_info (strictly json with case sensitive keys - ['name', 'designation', 'expertise']), customer_sentiment, minutes_of_meeting (Elaborate as much as possible. Try to keep in chronological order), customer_company_name_with_appointment_datetime_with_specialist_name Eg {{\"summary\": \"\", \"customer_info\": \"\", \"specialist_info\": \"\", \"customer_sentiment\": \"\", \"minutes_of_meeting\": \"\", \"customer_company_name_with_appointment_datetime_with_specialist_name\": \"\"}}:\nMessages:\n"
    for message in messages:
        prompt += f"Role: {message['role']}, Content: {message['content']}\n"

    # Generate the summary using the language model
    response = llm.invoke([SystemMessage(content=prompt)])
    print("Generated Summary:", response.content)
    json_resp = response.content.strip()
    if json_resp.startswith("```") and json_resp.endswith("```"):
        json_resp = json_resp[3:-3].strip()
    if json_resp.startswith("json"):
        json_resp = json_resp[4:].strip()

    # Parse LLM response
    try:
        return ast.literal_eval(json_resp)
    except (ValueError, SyntaxError) as e:
        print("Error parsing LLM response:", e)
        return {
            "summary": "",
            "customer_info": "",
            "customer_company_name_with_appointment_datetime_with_specialist_name": "",
            "specialist_info": "",
            "customer_sentiment": "",
            "minutes_of_meeting": "",
            "error": "Failed to parse summary from LLM response"
        }


def get_embedding_function() -> Embeddings:
    """Initialize and return an embedding model for text vectorization.

    Provider priority (mirrors get_llm):
      1. OpenAI  – when OPENAI_API_KEY is present in environment
      2. Google  – when GOOGLE_API_KEY is present in environment
      3. Ollama  – local fallback (requires Ollama running locally)

    Returns:
        Embeddings: Initialized embedding model instance
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {"model": "text-embedding-3-small"}
        if os.environ.get("OPEN_API_URL"):
            kwargs["base_url"] = os.environ["OPEN_API_URL"]
        print("\n** Initialized OpenAI Embeddings **\n")
        return OpenAIEmbeddings(**kwargs)

    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("\n** Initialized Google GenAI Embeddings **\n")
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
        )

    # Local fallback — requires Ollama running
    print("\n** Initialized Ollama Embeddings (local) **\n")
    return OllamaEmbeddings(model="nomic-embed-text")


class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, SystemMessage, AIMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs,
            }
        return super().default(obj)
