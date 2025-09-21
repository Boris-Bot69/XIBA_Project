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
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")  

if config.get("LANGSMITH_TRACING", True) in [True, "true", "True"]:
    os.environ["LANGSMITH_ENDPOINT"] = config.get("LANGSMITH_ENDPOINT", "")
    os.environ["LANGSMITH_API_KEY"] = config.get("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = config.get("LANGSMITH_PROJECT", "")
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
          llm = ChatOpenAI(model="gpt-4", temperature=0.0)
          # or
          llm = init_chat_model("gpt-4", model_provider="openai", temperature=0.0)
          ```
        
        - Google GenAI:
          ```
          # Required: pip install langchain-google-genai
          from langchain_google_genai import ChatGoogleGenerativeAI
          llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
          # or
          llm = init_chat_model("gemini-pro", model_provider="google_genai", temperature=0.0)
          ```
    
    Returns:
        BaseChatModel: Initialized language model instance
    
    Raises:
        ValueError: If required environment variables are missing
    """
    if "OPENAI_API_KEY" in config and config["OPENAI_API_KEY"]:

        open_ai_key = config["OPENAI_API_KEY"]
        if open_ai_key:
            os.environ["OPENAI_API_KEY"] = open_ai_key
            llm = ChatOpenAI(
                model="gpt-5-chat",
                temperature=0,
                max_tokens=None,
                timeout=None,
                base_url=config["OPEN_API_URL"],
                api_key=open_ai_key

                # max_retries=2,
                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                # base_url="...",
                # organization="...",
                # other params...
            )
            print ("\n** Initialized OpenAI LLM **\n")
            return llm

    if "GOOGLE_API_KEY" in config and config["GOOGLE_API_KEY"]:
        
        # Get API key from environment
        google_api_key = config["GOOGLE_API_KEY"]
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        os.environ["GOOGLE_API_KEY"] = google_api_key   
        print ("\n\ngoogle_api_key: ", google_api_key)     
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0
        )
        print ("\n** Initialized Google GenAI LLM **\n")
        return llm
    else: # return local model
        llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0.0)
        print ("\n** Initialized Ollama LLM **\n")
        return llm
    
class CustomChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI class to override the invoke method for tracing."""

    def invoke(self, messages, **kwargs):
        """Override invoke to add custom tracing logic."""
        # Custom tracing logic can be added here
        # print("Invoking CustomChatOpenAI with messages:", messages)
        return super().invoke(messages, **kwargs)    

def get_custom_llm() -> BaseChatModel:
    if "OPENAI_API_KEY" in config and config["OPENAI_API_KEY"]:

        open_ai_key = config["OPENAI_API_KEY"]
        if open_ai_key:
            os.environ["OPENAI_API_KEY"] = open_ai_key
            llm = CustomChatOpenAI(
                model="gpt-5-chat",
                temperature=0,
                max_tokens=None,
                timeout=None,
                base_url=config["OPEN_API_URL"],
                api_key=open_ai_key

                # max_retries=2,
                # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
                # base_url="...",
                # organization="...",
                # other params...
            )
            print ("\n** Initialized OpenAI LLM **\n")
            return llm
    else:

        google_api_key = config["GOOGLE_API_KEY"]
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        os.environ["GOOGLE_API_KEY"] = google_api_key   
        print ("\n\ngoogle_api_key: ", google_api_key)     
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0
        )
        print ("\n** Initialized Google GenAI LLM **\n")
        return llm
    
def summarize_conversation(messages: list) -> str:
    """Summarizes a conversation using a language model.

    Args:
        messages (list): A list of messages in the conversation.

    Returns:
        str: A summary of the conversation.
    """
    llm = get_custom_llm()  # Or get_llm() if you don't need custom tracing

    # Construct a prompt for summarization
    prompt = f"Please summarize the following conversation and provide a json format of summary, customer_info (strictly json with case sensitive keys - ['name', 'company', 'domain', 'email', 'topic']), specialist_info (strictly json with case sensitive keys - ['name', 'designation', 'expertise']), customer_sentiment, minutes_of_meeting (Elaborate as much as possible. Try to keep in chronological order), customer_company_name_with_appointment_datetime_with_specialist_name Eg {{\"summary\": \"\", \"customer_info\": \"\", \"specialist_info\": \"\", \"customer_sentiment\": \"\", \"minutes_of_meeting\": \"\", \"customer_company_name_with_appointment_datetime_with_specialist_name\": \"\"}}:\nMessages:\n"
    for message in messages:
        prompt += f"Role: {message['role']}, Content: {message['content']}\n"
    # prompt += "\nSummary:"

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
    return response.content.strip()


def get_embedding_function() -> Embeddings:
    """Initialize and return an embedding model for text vectorization.
    
    This function sets up an embedding model for converting text into vector
    representations. It supports multiple providers:
    
    - Ollama (local) embeddings
    - OpenAI embeddings
    - Other LangChain-compatible embedding models
    
    Configuration Options:
        - OpenAI:
          ```
          # Required: pip install langchain-openai
          from langchain_openai import OpenAIEmbeddings
          embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
          ```
        
        - Ollama (default):
          ```
          from langchain_ollama import OllamaEmbeddings
          embeddings = OllamaEmbeddings(model="nomic-embed-text")
          ```
    
    Returns:
        Embeddings: Initialized embedding model instance
    
    Note:
        The default implementation uses Ollama's nomic-embed-text model,
        which runs locally and doesn't require API keys.
    """
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