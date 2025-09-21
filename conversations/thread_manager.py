# conversations/thread_manager.py
from threading import Lock
from typing import Dict, List, Any
from uuid import uuid4

class ConversationManager:
    # ToDo: integrate with redis or any db for persistence
    
    """
    ConversationManager manages the state and history of multiple conversation threads  .
    Implements a singleton pattern to ensure consistent management across the app."""
    _instance = None
    _lock = Lock()  # Thread safety

    def __new__(cls, *args, **kwargs):
        with cls._lock:  # Prevent race conditions
            if cls._instance is None:
                cls._instance = super(ConversationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # conversation history will have thread_id: {"messages": [...]}
        self.conversation_history = {}

    def get_session(self, thread_id: str) -> Dict[str, Any]:
        """Get or create a session for a given thread ID."""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        return {
            "thread_id": thread_id, 
            "thread_name": "New Conversation",
            "messages": self.conversation_history[thread_id]
        }

    def update_thread_name(self, thread_id: str, name: str) -> None:
        """Update the name of a conversation thread."""
        # For simplicity, we just store the name in the first message
        if thread_id in self.conversation_history and self.conversation_history[thread_id]:
            self.conversation_history[thread_id][0]['thread_name'] = name
        elif thread_id in self.conversation_history:
            self.conversation_history[thread_id].append({'thread_name': name})

    def add_message(self, thread_id: str, data: Dict[str, Any]) -> None:
        """Store conversation messages."""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
        self.conversation_history[thread_id].append(data)

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Return conversation history."""
        return self.conversation_history.get(thread_id, [])
    
    def generate_thread_name(self, user_msg, final_msg):
        return "New Conversation - %s" % str(uuid4())
