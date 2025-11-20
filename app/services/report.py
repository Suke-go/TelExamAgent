from openai import AsyncOpenAI
from typing import Dict, List
import json
from datetime import datetime

from app.models.session import ConversationSession

REPORT_PROMPT = """
会話ログを分析して、定期健診レポートを以下の形式のJSONで生成してください。

以下の項目について、会話から抽出できる情報を整理してください：
1. 最近の体調の変化（めまい、喉の渇き、倦怠感など）
2. 食事の状況（甘いものを食べ過ぎていないか、規則正しく食べているか）
3. 運動の状況（散歩などはしているか）
4. 薬の服用状況（飲み忘れはないか）
5. 次回の来院予定の確認

JSON形式：
{
  "summary": "レポートの要約（2-3文）",
  "health_changes": {
    "status": "良好/注意/要観察",
    "details": "具体的な内容"
  },
  "diet": {
    "status": "良好/注意/要観察",
    "details": "具体的な内容"
  },
  "exercise": {
    "status": "良好/注意/要観察",
    "details": "具体的な内容"
  },
  "medication": {
    "status": "良好/注意/要観察",
    "details": "具体的な内容"
  },
  "next_visit": {
    "confirmed": true/false,
    "details": "次回来院に関する情報"
  },
  "concerns": ["懸念事項1", "懸念事項2"],
  "recommendations": ["推奨事項1", "推奨事項2"]
}

会話から情報が得られない項目は、その旨を明記してください。
JSONのみを返してください。他のテキストは含めないでください。
"""

class ReportService:
    """レポート生成サービス"""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate_report(self, session: ConversationSession) -> Dict:
        """会話セッションからレポートを生成"""
        # 会話ログをテキストに変換
        conversation_text = "\n".join([
            f"{'患者' if msg.role == 'user' else 'AI'}: {msg.content}"
            for msg in session.messages
            if msg.role != "system"
        ])
        
        if not conversation_text.strip():
            # 会話がない場合はデフォルトレポート
            return self._default_report()
        
        try:
            # LLMでレポート生成
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": REPORT_PROMPT},
                    {"role": "user", "content": f"会話ログ:\n{conversation_text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # JSONをパース
            report_text = response.choices[0].message.content.strip()
            # JSONのコードブロックを除去
            if report_text.startswith("```json"):
                report_text = report_text[7:]
            if report_text.startswith("```"):
                report_text = report_text[3:]
            if report_text.endswith("```"):
                report_text = report_text[:-3]
            report_text = report_text.strip()
            
            report = json.loads(report_text)
            
            # メタデータを追加
            report["session_id"] = session.session_id
            report["generated_at"] = datetime.now().isoformat()
            report["examination_date"] = session.start_time.isoformat()
            report["duration_minutes"] = (
                (session.end_time - session.start_time).total_seconds() / 60
                if session.end_time else None
            )
            
            return report
            
        except Exception as e:
            print(f"Error generating report: {e}")
            # エラー時はデフォルトレポートを返す
            report = self._default_report()
            report["error"] = str(e)
            return report
    
    def _default_report(self) -> Dict:
        """デフォルトレポート"""
        return {
            "summary": "会話データが不足しているため、レポートを生成できませんでした。",
            "health_changes": {
                "status": "不明",
                "details": "情報が不足しています"
            },
            "diet": {
                "status": "不明",
                "details": "情報が不足しています"
            },
            "exercise": {
                "status": "不明",
                "details": "情報が不足しています"
            },
            "medication": {
                "status": "不明",
                "details": "情報が不足しています"
            },
            "next_visit": {
                "confirmed": False,
                "details": "情報が不足しています"
            },
            "concerns": [],
            "recommendations": [],
            "generated_at": datetime.now().isoformat()
        }

