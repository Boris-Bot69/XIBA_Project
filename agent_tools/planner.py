from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import os
from pydantic import BaseModel, Field, SecretStr
# Import standard datetime
from datetime import datetime
from typing import Annotated, Dict, List, Any, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


from agent_tools.appointment.appointment_tools import book_appointment, check_appointment_availability
from agent_tools.case_studies.case_studies_tools import case_studies_tool
from agent_tools.customers.customer_tools import onboard_customer
from agent_tools.specialists.specialist_tools import get_specialist_availability
from prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_OUTPUT_INSTRUCTIONS


from llm_utils import get_llm
from utils import get_current_datetime_str


class ToolAData(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier")
    name: Optional[str] = Field(description="Name Field")
    description: Optional[str] = Field(description="Description Field")

class ToolADetail(BaseModel):
    data: Optional[ToolAData] = Field(
        default_factory=lambda: ToolAData(
            id=None,
            name=None,
            description=None,
        ),
        description="Tool A data"
    )

class ToolSelections(BaseModel):
    """Structured selections from different tools."""
    toolA_data: Optional[ToolADetail] = Field(
        None, description="Ranked Data"
    )


class PlannerResponse(BaseModel):
    """Schema for the holiday planner structured response."""
    response_type: Literal["conversation", "toolResults"] = Field(
        ..., description="Type of response: 'conversation' or 'toolResults'"
    )
    conversation_message: str = Field(
        ..., description="Message to display to the user"
    )
    detail: Optional[ToolSelections] = Field(
        None, description="Structured selections from tools (null for conversation type)"
    )

model = get_llm()

class PlannerState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# Define the list of tools available to this agent
tools = [
    case_studies_tool,
    onboard_customer,
    get_specialist_availability,
    book_appointment,
    check_appointment_availability,
]

# Get current datetime manually
current_datetime_str = get_current_datetime_str()

# Format the main system prompt (focuses on process, persona)
prompt = PLANNER_SYSTEM_PROMPT.replace("@@@current_datetime@@@", current_datetime_str)

# Define SIMPLIFIED response format instructions for the final structured output call.
# Focuses on schema population based on prior analysis (guided by main system prompt).

# Create the response_format tuple as per LangGraph documentation
# (simplified_prompt_for_final_call, schema_class)

response_format_config = (PLANNER_OUTPUT_INSTRUCTIONS, PlannerResponse)

# Create the React agent
# redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

# checkpointer = RedisCheckpoint(redis_url=redis_url)

planner_graph = create_react_agent(
    model,
    tools=tools,
    prompt=prompt,  # Main system prompt for ReAct loop
    # response_format=response_format_config  # Tuple for customized final structured output call
    # checkpointer=checkpointer,
    # should_check_point=True,
)
