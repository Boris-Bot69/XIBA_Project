import os
import csv
import json
import random
from typing import List
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from llm_utils import get_custom_llm
from utils import get_cwd

gllm = get_custom_llm()

with open(get_cwd() + os.sep + 'data' + os.sep + 'specialists.json', 'r') as f:
    specialists_data = json.load(f)

def llm_specialist_search(search_query: str, specialists: List[dict]) -> List[dict]:
    """Use LLM to intelligently match specialists based on search query and return 1 specialist json"""
    try:
        # Prepare specialist information for LLM
        specialist_info = []
        for specialist in specialists:
            specialist_str = f"ID: {specialist['specialist_id']}, Name: {specialist['name']}, Title: {specialist['title']}, Product: {specialist['products']}, Skills: {specialist['skills']}, Integrations: {specialist['integrations']}, Domain: {specialist['industries']}"
            specialist_info.append(specialist_str)

        specialist_list_text = "\n".join(specialist_info)
        import ast  # For safely evaluating LLM output
        # Create LLM prompt
        system_prompt = """You are a technical specialist matching domain expert. Analyze the user query and match it with the most appropriate specialist based on their specialties, sub-specialties, and expertise. Return json dict of matched specialist with full json structure. If no match, return empty list []."""

        human_prompt = f"""Specialist Query: "{search_query}"

Available Specialists:
{specialist_list_text}

Analyze the user query and match with the most relevant specialists based on domain and expertise. Consider:
1. Specialty alignment with domain, expertise and skills.

Filter specialists and return 1 best match"""

        # Call LLM
        response = gllm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        # print("\nSpecialist LLM response:", response)    
        json_resp = response.content.strip()
        if json_resp.startswith("```") and json_resp.endswith("```"):
            json_resp = json_resp[3:-3].strip()
        if json_resp.startswith("json"):
            json_resp = json_resp[4:].strip()

        # Parse LLM response
        return ast.literal_eval(json_resp)

    except Exception as e:
        print(f"LLM search error: {e}")
        # Fallback to simple keyword matching
        return [specialists[0]] if specialists else []


@tool
def get_specialist_availability(search_query: str) -> dict:
    """Use LLM to intelligently match specialists based on search query and return 1 specialist based on domain and expertise"""
    try:
        specialist_data = llm_specialist_search(search_query, specialists_data)
        # specialist_data = random.choice(specialists_data)
        if "matched_specialist" in specialist_data:
            specialist_data = specialist_data["matched_specialist"]
        print ("\n\nSpecialist 1st matched data >>>> ", specialist_data)
        if not specialist_data:
            specialist_data = specialists_data[0]
        # print ("\n\nSpecialist matched data >>>> ", specialist_data)
        return {"message": f"Specialist selected: {str(specialist_data)}", "specialist_id": specialist_data.get("specialist_id", ""), "name": specialist_data.get("name", "")}
    except Exception as e:
        print(f"Error fetching specialist availability: {e}")
        return {"error": f"Error fetching specialist availability: {str(e)}"}
