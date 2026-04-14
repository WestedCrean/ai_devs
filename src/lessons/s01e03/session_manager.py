from .prompts import get_system_prompt

from typing import Dict, List
import threading


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[Dict]] = {}
        self.lock = threading.Lock()

    def get_session(self, session_id: str) -> List[Dict]:
        """Get existing session or create new one"""
        with self.lock:
            if session_id not in self.sessions:
                # Initialize with system prompt
                self.sessions[session_id] = [
                    {"role": "system", "content": get_system_prompt()}
                ]
            return self.sessions[session_id]

    def update_session(self, session_id: str, messages: List[Dict]):
        """Update session messages"""
        with self.lock:
            self.sessions[session_id] = messages
