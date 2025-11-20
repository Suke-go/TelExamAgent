from datetime import datetime
from typing import List, Dict, Optional
import uuid
import json

class ConversationMessage:
    """会話のメッセージ"""
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

class ConversationSession:
    """会話セッション"""
    def __init__(self, session_id: Optional[str] = None, phone_number: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.phone_number = phone_number
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.messages: List[ConversationMessage] = []
    
    def add_message(self, role: str, content: str):
        """メッセージを追加"""
        self.messages.append(ConversationMessage(role, content))
    
    def finish(self):
        """セッションを終了"""
        self.end_time = datetime.now()
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "phone_number": self.phone_number,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "messages": [msg.to_dict() for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        session = cls(
            session_id=data["session_id"],
            phone_number=data.get("phone_number")
        )
        session.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            session.end_time = datetime.fromisoformat(data["end_time"])
        session.messages = [ConversationMessage.from_dict(msg) for msg in data.get("messages", [])]
        return session

