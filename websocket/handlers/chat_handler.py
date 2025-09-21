import asyncio
import time
import traceback
from uuid import uuid4
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, cast
from fastapi import WebSocketDisconnect
from websocket import WebSocket
from conversations.thread_manager import ConversationManager
from utils import chunk_response, _ensure_serializable, get_redis_instance, safe_jsondumps
# from llm_utils import MessageEncoder
from websocket.manager import WebSocketManager
# Import necessary types for checkpointing and streaming
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.runnables import RunnableConfig

from agent_tools.planner import planner_graph
from app_logger import log_message, logger
from llm_utils import summarize_conversation

ws_manager = WebSocketManager()
conversation_mgr = ConversationManager()
redis_client = get_redis_instance()



async def _process_graph_stream(
    thread_id: str, user_id: str, initial_state: Dict[str, Any], config: RunnableConfig
) -> None:
    """
    Processes the Thread graph using astream_events, handling
    token streaming, event reporting, and session updates from a single invocation.
    """
    is_streaming_tokens = False
    latest_full_content = ""
    current_message_tokens: List[str] = []
    seen_events: Set[str] = set()
    sent_tool_results: Set[str] = set()
    tools_started: Set[str] = set()
    tool_data: Dict[str, Dict[str, Any]] = {}  # Type hint added
    final_ai_message_content: Optional[str] = None
    structured_response_sent = False

    # Config for event streaming
    # Combine original config with stream_mode
    event_stream_config: RunnableConfig = {
        **config,
        # Recursion limit might be needed for complex graphs
        # "recursion_limit": 25
    }
    # stream_mode is passed via kwargs now
    stream_kwargs = {"stream_mode": "events"}

    try:
        logger.info(f"Starting unified event stream for thread {thread_id} with config: {event_stream_config}")

        # Stream events from the graph
        async for event in planner_graph.astream_events(initial_state, config=event_stream_config, version="v2", **stream_kwargs):
            event_type = event.get("event")
            event_data = event.get("data", {})
            event_name = event.get("name", "")  # Often the node name
            run_id = event.get("run_id")  # Useful for tracking
            tags = event.get("tags", [])  # Can contain tool names etc.
            logger.info( f"Event received: type={event_type}, name={event_name}, run_id={run_id}")
            # Skip certain events if needed           
            if event_name == "ChatGoogleGenerativeAI" or event_name == "CustomChatOpenAI":
                continue
            # logger.debug(f"Event data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'N/A'}")
            # print("\n\n>>>Stream event: %s" % safe_jsondumps(_ensure_serializable(event), indent=2))
            # --- 1. Token Streaming Logic ---
            # Look for chunks from the language model stream
            # Check both 'on_chat_model_stream' and 'on_llm_stream' for compatibility
            if event_type in ["on_chat_model_stream", "on_llm_stream"]:
                chunk = event_data.get("chunk")
                if chunk:
                    # Extract content based on expected chunk structure (e.g., AIMessageChunk)
                    content_piece = None
                    if hasattr(chunk, 'content'):
                        content_piece = chunk.content
                    elif isinstance(chunk, str):  # Simpler LLM might just yield strings
                        content_piece = chunk

                    if content_piece:
                        # Start streaming if this is the first token
                        if not is_streaming_tokens:
                            is_streaming_tokens = True
                            current_message_tokens = []
                            await ws_manager.send_message(
                                thread_id,
                                {"type": "msg_stream_start", "timestamp": datetime.now().isoformat()}
                            )
                            logger.info(f"Started token stream for thread {thread_id}")
                            last_logged_length = 0  # Track length for progress logging
                            log_interval = 100  # Log every 100 characters

                        # Send the token
                        current_message_tokens.append(content_piece)
                        latest_full_content = "".join(current_message_tokens)  # Keep track of the full message
                        await ws_manager.send_message(
                            thread_id,
                            {
                                "type": "msg_stream",
                                "message": content_piece,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        # Log progress periodically based on length
                        current_length = len(latest_full_content)
                        if current_length >= last_logged_length + log_interval:
                            logger.debug(f"Streaming progress: ~{current_length} characters...")
                            last_logged_length = current_length
                        # logger.debug(f"Sent token: '{content_piece}'") # Removed log-per-token

            # --- 2. Structured Response & Final Message Logic ---
            # Often the final AI response comes in on_chain_end or node_end events data
            # Specifically look at the end of the main planner node if applicable
            # Or check for specific keys like 'final_output' or 'structured_response' if your graph sets them
            if event_type == "on_chain_end" or event_type == "on_node_end":
                # Check if the event data contains the final message(s)
                output = event_data.get("output")  # Output is common place for final result
                if output:
                    # Output structure can vary greatly based on graph/node implementation
                    # Example: if output is the final state dictionary
                    if isinstance(output, dict) and "messages" in output:
                        final_messages = output["messages"]
                        if final_messages and isinstance(final_messages, list):
                            # Get the *last* AI message as the definitive response
                            for msg in reversed(final_messages):
                                role = None
                                content = None
                                if isinstance(msg, dict):
                                    role = msg.get("role")
                                    content = msg.get("content")
                                elif hasattr(msg, "type") and hasattr(msg, "content"):
                                    role = getattr(msg, "type", None) or getattr(msg, "role", None)
                                    content = getattr(msg, "content", None)

                                if (role == "ai" or role == "assistant") and content:
                                    final_ai_message_content = content
                                    # Check content is not None before len()
                                    # content_len = len(final_ai_message_content) if final_ai_message_content is not None else 0
                                    # logger.info(f"Captured final AI message content (length {content_len}) from event {event_type}")
                                    break  # Stop after finding the last AI message

                    # Example: Check for a specific structured response key
                    elif isinstance(output, dict) and "structured_response" in output and not structured_response_sent:
                        structured_data = _ensure_serializable(output["structured_response"])
                        await ws_manager.send_message(
                            thread_id,
                            {
                                "type": "structured_response",
                                "data": structured_data,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        structured_response_sent = True
                        logger.info("Sent structured response data to client from event")

            # --- 3. High-Level Agent Event Logic ---
            node_name = event_name  # Use event name as node name by default
            display_name = node_name.replace("_", " ").title() if node_name else "Graph"
            event_id = f"{event_type}:{node_name}:{run_id}"  # More unique event ID

            # Avoid sending duplicate status events
            if event_id in seen_events:
                continue

            # Map event types to our desired frontend types
            normalized_event_type: Optional[str] = None
            message: Optional[str] = None
            tool_info: Dict[str, Any] = {}

            # Check event_type is not None before using 'in'
            is_tool_event = event_type is not None and ("tool" in event_type or any(tag.startswith("tool:") for tag in tags))

            if event_type == "on_node_start" and not is_tool_event:
                normalized_event_type = "on_node_start"
                message = await _format_event_message(normalized_event_type, cast(Dict[str, Any], event_data), node_name, thread_id)
            elif event_type == "on_node_end" and not is_tool_event:
                normalized_event_type = "on_node_end"
                message = await _format_event_message(normalized_event_type, cast(Dict[str, Any], event_data), node_name, thread_id)
            elif event_type == "on_tool_start":
                normalized_event_type = "on_tool_start"
                tool_input = event_data.get("input", {})  # Tool input often here
                tool_info["name"] = node_name
                input_str = str(tool_input)
                tool_info["input"] = input_str[:100] + "..." if len(input_str) > 100 else input_str
                message = await _format_event_message(normalized_event_type, cast(Dict[str, Any], event_data), node_name, thread_id)
                tools_started.add(node_name)
                logger.info(f"Marked tool {node_name} as started")
                # Initialize tool data storage
                if node_name not in tool_data:
                    tool_data[node_name] = {"input": tool_input}
            elif event_type == "on_tool_end":
                normalized_event_type = "on_tool_end"
                tool_output = event_data.get("output", {})  # Tool output often here
                tool_info["name"] = node_name
                output_str = str(tool_output)
                tool_info["output"] = output_str[:100] + "..." if len(output_str) > 100 else output_str
                message = await _format_event_message(normalized_event_type, cast(Dict[str, Any], event_data), node_name, thread_id)
                print ("TTTTTTTTTTTTTT EEEEEEEEE DDDDDDDDDDD message >>>", message)
                # Store tool output
                if node_name in tool_data:
                    tool_data[node_name]["output"] = tool_output
                else:  # Should have started first, but handle defensively
                    tool_data[node_name] = {"output": tool_output}

            # logger.info( f"Message ={str(message)}\n")
            # Send the formatted agent event if relevant
            if normalized_event_type and message:
                seen_events.add(event_id)  # Mark as seen only if sending
                response = {
                    "type": "agent_event",
                    "event_type": normalized_event_type,
                    "node_name": node_name,
                    "display_name": display_name,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "event_order": len(seen_events)
                }
                if tool_info:
                    response["tool_info"] = tool_info

                logger.info(f"Sending agent event: {normalized_event_type} - {message[:50]}...")
                await ws_manager.send_message(thread_id, response)

        # --- Stream Processing Finished ---
        logger.info(f"Graph event stream finished for thread {thread_id}")
        # import pdb; pdb.set_trace()
        # Send token stream end if we were streaming
        if is_streaming_tokens:
            await ws_manager.send_message(
                thread_id,
                {"type": "msg_stream_end", "timestamp": datetime.now().isoformat()}
            )
            logger.info(f"Token stream ended for thread {thread_id}")

        # Determine the final AI message: prefer explicitly captured content, fall back to accumulated tokens
        final_content_to_save = final_ai_message_content if final_ai_message_content is not None else latest_full_content

        # Update session with the final AI message - THIS IS NOW LESS CRITICAL
        # The session cache in memory/Redis is mainly for quick state passing to the graph.
        # The DB is the source of truth for history.
        session = conversation_mgr.get_session(thread_id)
        if session:
            # --- Save AI Message to DB --- (Phase 1 - Updated)
            if final_content_to_save:
                current_thread_name = session.get("thread_name", "New Conversation")
                current_time = datetime.now()
                message_id = str(uuid4())
                
                flight_cards = [] # self.generate_flight_cards()
                hotel_cards = [] # self.generate_hotel_cards()

                # Save the message to database with cards if available
                conversation_mgr.add_message(thread_id, {
                    "user_id": user_id,
                    "role": "ai",
                    "content": final_content_to_save,
                    "cards": [],
                    "id": message_id,
                    "timestamp": current_time
                })

                logger.info(f"Saved final AI message to DB for thread {thread_id}")
                
                # --- Thread Naming Logic (Phase 3) ---
                # Check if this thread still has the default name
                if current_thread_name == "New Conversation":
                    try:
                        # Get the user's message that initiated this response
                        messages = session.get("messages", [])
                        logger.info(f"Session has {len(messages)} messages for thread naming")
                        user_message = ""
                        for msg in messages:
                            if msg.get("role") == "human":
                                user_message = msg.get("content", "")
                                break
                        
                        if user_message and final_content_to_save:
                            logger.info(f"Generating name for thread {thread_id} - User message: {user_message[:30]}...")
                            # Generate a thread name based on the conversation
                            generated_name = await conversation_mgr.generate_thread_name(
                                user_message=user_message,
                                ai_message=final_content_to_save
                            )
                            
                            # Update thread name in storage system
                            if generated_name != "New Conversation" and generated_name.strip():
                                # Update in thread-based storage
                                thread_updated = await conversation_mgr.update_thread_name(
                                    thread_id=thread_id,
                                    new_name=generated_name
                                )
                                
                                if thread_updated:
                                    # Update the session's thread name
                                    session["thread_name"] = generated_name
                                    logger.info(f"Thread {thread_id} renamed to: {generated_name}")
                                    
                                    # Send thread name update to client
                                    await ws_manager.send_message(
                                        thread_id,
                                        {
                                            "type": "thread_name_updated",
                                            "thread_id": thread_id,
                                            "name": generated_name,
                                            "timestamp": datetime.now().isoformat(),
                                        }
                                    )
                    except Exception as naming_err:
                        logger.error(f"Error updating thread name: {naming_err}")
                        # Continue with default name if naming fails
                # else - Thread already has a custom name, keep using it
                # -------------------------------

                # Ensure proper session message management for Redis
                if "messages" not in session:
                    session["messages"] = []

                if final_content_to_save:
                    # Add the AI message to the session
                    ai_message = {
                        "id": message_id,
                        "role": "ai",
                        "content": final_content_to_save,
                        "timestamp": current_time.isoformat()
                    }
                    conversation_mgr.add_message(thread_id, ai_message)  # Also add to in-memory history
                        
                    logger.info(f"Added AI message to session. Session now has {len(session['messages'])} messages")
                elif not any(msg.get("role") == "ai" for msg in session["messages"][1:]):  # Check if *any* AI response was added after the first user msg
                    fallback_response = "I'm sorry, I wasn't able to generate a response. How else can I help you with your travel plans?"
                    logger.warning("No AI content generated, adding fallback response to session cache.")
                    conversation_mgr.add_message(thread_id, {"role": "ai", "content": fallback_response})
                    final_content_to_save = fallback_response

                # Send the complete agent response (for non-streaming clients or summary)
                if final_content_to_save:
                    response_message = {
                        "type": "agent_response",
                        "content": final_content_to_save,
                        "agent": "planner",
                        "timestamp": datetime.now().isoformat(),
                    }
                                            
                    await ws_manager.send_message(
                        thread_id,
                        response_message
                    )
                    logger.info("Sent complete agent response message")
                else:
                    # This case should ideally be covered by the fallback logic above
                    logger.warning("No final AI content available to send as agent_response.")

        # Send completion message
        await ws_manager.send_message(
            thread_id,
            {
                "type": "completed",
                "thread_id": thread_id,
                "agent": "planner",
                "timestamp": datetime.now().isoformat(),
            }
        )
        
    except asyncio.CancelledError:
        logger.info(f"Thread processing task cancelled for {thread_id}")
        # Send token stream end if cancelled mid-stream
        if is_streaming_tokens:
            try:
                await ws_manager.send_message(
                    thread_id,
                    {"type": "msg_stream_end", "timestamp": datetime.now().isoformat()}
                )
            except Exception:
                pass  # Ignore errors during cancellation cleanup
        raise  # Re-raise cancellation

    except Exception as e:
        logger.error(f"Error during Thread stream processing for {thread_id}: {e}")
        logger.error(f"Stream processing traceback: {traceback.format_exc()}")

        # Send error to client
        try:
            await ws_manager.send_message(
                thread_id,
                {
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as ws_err:
            logger.error(f"Failed to send error to WebSocket: {ws_err}")

        # Add fallback response to session
        fallback_response = "I apologize, but I encountered an error while processing your request. How else can I help you with your travel plans?"
        session = conversation_mgr.get_session(thread_id)
        if session:
            if "messages" not in session:
                session["messages"] = []
            # Avoid adding duplicate fallbacks if error happens late
            if not session["messages"] or session["messages"][-1].get("content") != fallback_response:
                conversation_mgr.add_message(thread_id, {"role": "ai", "content": fallback_response})
                # Attempt to send fallback as agent response
                try:
                    fallback_message = {
                        "type": "agent_response",
                        "content": fallback_response,
                        "agent": "planner",
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    await ws_manager.send_message(
                        thread_id,
                        fallback_message
                    )
                    logger.info("Sent error fallback response")
                except Exception as ws_err:
                    logger.error(f"Failed to send fallback response to WebSocket: {ws_err}")
    finally:
        # Ensure completion message is sent even if errors occurred before it
        was_cancelled = False
        try:
            # Send completed unless the task was cancelled (handled by re-raising CancelledError)
            await ws_manager.send_message(
                thread_id,
                {
                    "type": "completed",
                    "thread_id": thread_id,
                    "agent": "planner",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            logger.debug("Sent completion message in finally block")

        except asyncio.CancelledError:  # Catch if cancellation happens here
            was_cancelled = True  # Mark as cancelled to avoid logging error below
            raise  # Re-raise immediately
        except Exception as final_ws_err:
            # Only log error if not cancelled
            if not was_cancelled:
                logger.error(f"Error sending completion message in finally block: {final_ws_err}")

async def _format_event_message(event_type: str, event_data: Dict[str, Any], node_name: str = "", thread_id: str = "") -> Optional[str]:
    """
    Maps key LangGraph events to user-facing messages for the Thread agent.
    (Retained from original implementation)
    """
    print ("\nevent_type>>>%s<<< " % event_type, "node_name>>>%s<<<" % node_name)
    if not node_name:
        # Use default message for generic graph start/end if node name is missing
        if event_type == "on_node_start":
            return "Starting holiday plan..."
        # if event_type == "on_node_end": return "Finished step." # Usually too noisy
        return None


    if node_name == "book_appointment" and event_type == "on_tool_end":

        # store in redis
        try:
            print ("\\nevent_data >>> ", thread_id, event_data)
            messages_for_graph = conversation_mgr.get_history(thread_id)
            all_messages = _ensure_serializable(messages_for_graph)
            # print ("all_messages >>> ", all_messages)
            summary = summarize_conversation(all_messages)
            # print ("summary 2222 >>>> ", summary)
            if not summary or "error" in summary:
                summary = "No summary available."
            else:
                summary["thread_id"] = thread_id
                redis_client.hset(f"leads_generated", summary.get("customer_company_name_with_appointment_datetime_with_specialist_name", thread_id), json.dumps(summary))

            print ("summary >>>> ", summary)
            redis_client.set(f"conversation:thread:{thread_id}", safe_jsondumps({"messages": _ensure_serializable(messages_for_graph)}))
        except Exception as e:
            print (f"Error saving conversation to Redis for thread: {e}")
            logger.error(f"Redis Save Error Traceback: {traceback.format_exc()}")

    node_lower = node_name.lower()

    # --- Node Start Events ---
    if event_type == "on_node_start":
        if "planner" in node_lower or "agent" in node_lower:  # Be more general
            return "Planning your perfect holiday experience... 🏝️"
        elif "fetch" in node_lower or "search" in node_lower:
            return f"Starting search: {node_name.replace('_', ' ').title()}..."  # Generic search start
        # Add more specific node starts if needed
        else:
            # Default start message for less critical nodes (or return None to hide)
            # return f"Starting step: {node_name.replace('_', ' ').title()}..."
            return None  # Keep UI cleaner

    # --- Tool Start Events ---
    elif event_type == "on_tool_start":
        # Tool names should be accurate from event_name
        if node_name == "search_doctors":
            return "Searching doctors based on your requirement.. 🔍"
        elif node_name == "calculate_doctor_match_score":
            return "Matching Doctor Score.."
        elif node_lower == "check_appointment_availability":
            return "Checking appointment availability.. 🏨"
        elif node_lower == "get_patient_issues":
            return "Fetching patient profile.. 🏨"
        elif node_lower == "patient_profile":
            return "Looking patient profile.. 🏨"
        elif node_lower == "get_patient_issues":
            return "Getting Patient Prior Issues.. 🏨"
        elif node_lower == "check_appointment_availability":
            return "Checking Appointment"
        elif node_lower == "book_appointment":
            return "Booking Appointment"
        

    # --- Tool End Events ---
    elif event_type == "on_tool_end":
        # Tool ends are often less informative for the user unless there's an error
            
        return None

    # --- Node End Events ---
    elif event_type == "on_node_end":
        if "planner" in node_lower or "agent" in node_lower:
            # Avoid generic end message if a final response is coming
            return "Syncing doctors and making appointment…"
            # return None
        else:
            # return f"Finished step: {node_name.replace('_', ' ').title()}." # Usually too noisy
            return None

    # --- Ignore all other event types for user-facing messages ---
    return None


async def handle(thread_id: str, user_id: str, user_input: Any):
    try:
        print(f"Received message on thread {thread_id}: {user_input}")
        if isinstance(user_input, dict):
            user_query = user_input.get("query", user_input.get("message", "")).strip()
        elif isinstance(user_input, str):
            user_query = user_input.strip()
        else:
            user_query = ""
        print ("user_query >>" , user_query)
        session = conversation_mgr.get_session(thread_id)
        if user_query:
            # Get agent response
            current_time = datetime.now()
            message_id = str(uuid4())
            user_message = {
                "id": message_id,
                "user_id": user_id,
                "role": "human",
                "content": user_query,
                "timestamp": current_time.isoformat()  # Store as ISO string for Redis
            }

            conversation_mgr.add_message(thread_id, user_message)
            messages_for_graph = conversation_mgr.get_history(thread_id) # this will have current message added above
            print(f"Stored user message in session for thread {thread_id}. Session now has {len(messages_for_graph)} messages.")


            # Create initial state for the graph using *these* messages
            initial_state = {"messages": messages_for_graph, "thread_id": thread_id, "user_id": user_id}



            # Configure graph invocation
            # Ensure thread_id is correctly placed within 'configurable'
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    # Add checkpoint namespace if needed, though auto-save handles this
                    # "checkpoint_ns": "planner",
                },
                # No longer need stream_mode or stream_tokens here, handled in astream_events
            }

            # Send "thinking" message
            await ws_manager.send_message(
                thread_id,
                {
                    "type": "processing",
                    "message": "VJ Bot thinking... 🤔",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            processing_task = asyncio.create_task(
                _process_graph_stream(thread_id, user_id, initial_state, config)
            )


            try:
                logger.debug(f"Waiting for Thread stream task for thread {thread_id}...")
                await processing_task
                logger.info(f"Thread stream task completed for thread {thread_id}.")
            except asyncio.CancelledError:
                logger.info(f"Thread stream task cancelled for {thread_id}")
                raise
            except Exception as e:
                logger.error(f"Error awaiting Thread stream task for {thread_id}: {e}")
                logger.error(f"Stream/Save Error Traceback: {traceback.format_exc()}")
                # Send error to client (handled within _process_graph_stream's exception block)
            finally:
                logger.debug(f"Finished handle_message for thread {thread_id}")


            # response = await thread_manager.route_to_agents(thread_id, user_query)
            # print("\nFull response:", response, "\n\n")
            # # Stream response in chunks (for "typing" effect)
            # indx = 0
            # for chunk in chunk_response(response, size=10):
            #     if indx == 0:
            #         await ws_manager.send_message(thread_id, {"type": "msg_stream_start"})
            #     print("Sending chunk:", chunk)
            #     indx += 1
            #     await asyncio.sleep(0.2) # Simulate delay for streaming effect
            #     ret = await ws_manager.send_message(thread_id, {"message": chunk, "type": "msg_stream"})
            #     print("\tChunk sent, ret:", ret)
            # await ws_manager.send_message(thread_id, {"type": "msg_stream_end"})
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: chat/{thread_id}")  
        await ws_manager.disconnect(thread_id)
    except Exception as e:      
        print(f"Error in chat handler: {str(e)}")
        await ws_manager.send_message(thread_id, {
            "type": "error",
            "message": f"Error: {str(e)}"
        })
