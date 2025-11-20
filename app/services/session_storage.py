import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from app.models.session import ConversationSession

# データ保存ディレクトリ
DATA_DIR = Path("data")
SESSIONS_DIR = DATA_DIR / "sessions"

# ディレクトリを作成
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

class SessionStorage:
    """会話セッションの保存・読み込み"""
    
    @staticmethod
    def save_session(session: ConversationSession):
        """セッションを保存"""
        file_path = SESSIONS_DIR / f"{session.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Session saved: {session.session_id}")
    
    @staticmethod
    def load_session(session_id: str) -> Optional[ConversationSession]:
        """セッションを読み込み"""
        file_path = SESSIONS_DIR / f"{session_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ConversationSession.from_dict(data)
    
    @staticmethod
    def list_sessions(limit: int = 100) -> List[ConversationSession]:
        """セッション一覧を取得（新しい順）"""
        sessions = []
        for file_path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append(ConversationSession.from_dict(data))
                    if len(sessions) >= limit:
                        break
            except Exception as e:
                print(f"Error loading session {file_path}: {e}")
        return sessions

