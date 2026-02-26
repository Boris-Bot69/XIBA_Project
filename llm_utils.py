import ast
import os
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )


def get_custom_llm() -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )


def get_embedding_function() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )


def summarize_conversation(messages: list) -> str:
    llm = get_custom_llm()

    prompt = f"Please summarize the following conversation and provide a json format of summary, customer_info (strictly json with case sensitive keys - ['name', 'company', 'domain', 'email', 'topic']), specialist_info (strictly json with case sensitive keys - ['name', 'designation', 'expertise']), customer_sentiment, minutes_of_meeting (Elaborate as much as possible. Try to keep in chronological order), customer_company_name_with_appointment_datetime_with_specialist_name Eg {{\"summary\": \"\", \"customer_info\": \"\", \"specialist_info\": \"\", \"customer_sentiment\": \"\", \"minutes_of_meeting\": \"\", \"customer_company_name_with_appointment_datetime_with_specialist_name\": \"\"}}:\nMessages:\n"
    for message in messages:
        prompt += f"Role: {message['role']}, Content: {message['content']}\n"

    response = llm.invoke([SystemMessage(content=prompt)])
    json_resp = response.content.strip()
    if json_resp.startswith("```") and json_resp.endswith("```"):
        json_resp = json_resp[3:-3].strip()
    if json_resp.startswith("json"):
        json_resp = json_resp[4:].strip()

    try:
        return ast.literal_eval(json_resp)
    except (ValueError, SyntaxError):
        return {
            "summary": "",
            "customer_info": "",
            "customer_company_name_with_appointment_datetime_with_specialist_name": "",
            "specialist_info": "",
            "customer_sentiment": "",
            "minutes_of_meeting": "",
            "error": "Failed to parse summary from LLM response"
        }


class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (HumanMessage, SystemMessage, AIMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs,
            }
        return super().default(obj)
