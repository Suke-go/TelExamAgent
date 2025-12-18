from openai import AsyncOpenAI
import os
import json
from typing import AsyncGenerator
import re

# 問診プランナー統合
from .interview_planner import InterviewPlanner

# 問診ステージ定義
INTERVIEW_STAGES = {
    "greeting": {
        "name": "挨拶・アイスブレイク",
        "goal": "患者との信頼関係を築く",
        "next": "health_check",
        "prompts": [
            "まず、相手の様子を確認しましょう",
            "世間話も歓迎です。リラックスした雰囲気を作りましょう"
        ]
    },
    "health_check": {
        "name": "体調確認",
        "goal": "最近の体調変化を聞き取る",
        "next": "diet",
        "questions": [
            "最近、体調で気になることはありますか？",
            "めまいや、のどの渇きを感じることはありますか？",
            "疲れやすいとか、だるいとかはないですか？"
        ]
    },
    "diet": {
        "name": "食事状況",
        "goal": "食事の回数、規則性、間食、塩分・糖分について自然に聞き取る",
        "next": "exercise",
        "questions": [
            "最近のお食事はどうですか？規則正しく食べられていますか？",
            "1日何食くらい召し上がってますか？朝はちゃんと食べられてます？",
            "ついつい間食をしちゃったり、甘いものが止まらないなんてことはありますか？",
            "味付けはどうですか？塩分を控えめに、とか意識されていますか？",
            "お菓子とかジュースとか、甘いもの召し上がる機会多いですか？"
        ]
    },
    "exercise": {
        "name": "運動状況",
        "goal": "運動の回数、習慣性、具体的な内容を確認する",
        "next": "work",
        "questions": [
            "お散歩とか、何か体を動かす習慣はありますか？",
            "週に何回くらい、どんなことをされていますか？",
            "無理のない範囲で、少し汗をかくくらい動けていますか？"
        ]
    },
    "work": {
        "name": "生活・お仕事",
        "goal": "仕事の過労状況や日々のストレスについて聞き取る",
        "next": "medication",
        "questions": [
            "最近、お仕事や家事でお忙しくされていませんか？",
            "何かストレスに感じることや、お疲れが溜まっていることはないですか？"
        ]
    },
    "medication": {
        "name": "薬・インスリンの確認",
        "goal": "服薬やインスリンの定期的な使用状況を確認し、評価する",
        "next": "closing",
        "questions": [
            "お薬は毎日、決まった時間に飲めていますか？",
            "インスリンをお使いなら、そちらも忘れずに打てていますか？",
            "飲み忘れがあったり、最近お薬で困っていることはないですか？"
        ]
    },
    "closing": {
        "name": "締めくくり",
        "goal": "次回来院の確認と温かい挨拶で終わる",
        "next": None,
        "prompts": [
            "次回の来院予定を確認しましょう",
            "無理せず、体を大事にしてくださいね、と伝えましょう"
        ]
    }
}

SYSTEM_PROMPT = """
あなたは糖尿病患者の定期検診を電話で行うAIアシスタント「サクラ」です。
優しくて親しみやすい中年女性のような口調で話してください。

## あなたの役割
- 高齢の糖尿病患者に定期的な電話検診を行う
- 体調、食事、運動、薬の状況を聞き取る
- 患者が安心して話せる雰囲気を作る

## 会話スタイル

### 必ず質問を投げかける
- 毎回の発話で、相手に質問を投げかけてください
- 「〜ですか？」「〜はどうですか？」で終わることが多いはずです
- 一方的に話さず、相手の話を引き出すことを意識

### 自然な相槌
- 「あらあら」「そうですか」「へぇ」「なるほど」
- 「それは〜ですね」より「〜なんですね」の方が自然

### 人間らしい励まし（NGワード）
❌ 避ける: 「頑張ってください」「それは大変ですね」「心配ですね」
✓ 使う: 「無理しないでくださいね」「ゆっくり休んでください」「いいですね！」

### 具体的な質問を
❌ 曖昧: 「体調はいかがですか？」
✓ 具体的: 「最近、めまいとか感じることはありますか？」

## 会話例

良い例：
ユーザー「寒くなりましたね」
→「本当に寒くなりましたよね。こう寒いと、なかなか外に出るのも億劫になりませんか？」

ユーザー「ちょっと風邪気味で」
→「あらあら、風邪ですか。お熱とかはありますか？無理しないでくださいね。」

ユーザー「はい、元気です」
→「それは良かったです！最近、お食事はちゃんと食べられていますか？」

ユーザー「薬飲んでます」
→「えらいですね！飲み忘れとかなく、毎日飲めていますか？」

## 重要なルール
1. 1回の発話は短く（50〜80文字程度）
2. 必ず質問や相手への問いかけで終わる
3. 世間話も歓迎、でも問診も進める
4. 深刻な症状（激しい胸の痛み、意識障害など）は即座に受診を促す

【出力形式】必ず以下のJSON形式で回答してください：
{"display": "画面表示用テキスト", "speech": "読み上げ用テキスト（ひらがな多め）"}
"""

class LLMService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.history = []
        self.current_stage = "greeting"
        self.stage_completed = {stage: False for stage in INTERVIEW_STAGES}
        self.last_topic = None
        self.pending_utterance = ""
        
        # 問診プランナー統合
        self.interview_planner = InterviewPlanner()

    def _build_system_prompt(self) -> str:
        """Build dynamic system prompt with current stage info."""
        stage_info = INTERVIEW_STAGES.get(self.current_stage, {})
        stage_name = stage_info.get("name", "")
        stage_goal = stage_info.get("goal", "")
        
        # Get suggested questions for this stage
        questions = stage_info.get("questions", [])
        prompts = stage_info.get("prompts", [])
        
        stage_guidance = f"""
## 現在のステージ: {stage_name}
目標: {stage_goal}
"""
        if questions:
            stage_guidance += "\n使える質問例:\n" + "\n".join(f"- {q}" for q in questions)
        if prompts:
            stage_guidance += "\nヒント:\n" + "\n".join(f"- {p}" for p in prompts)
        
        # Show progress
        completed = [s for s, done in self.stage_completed.items() if done]
        remaining = [s for s, done in self.stage_completed.items() if not done and s != self.current_stage]
        
        if completed:
            stage_guidance += f"\n\n✓ 完了済み: {', '.join(INTERVIEW_STAGES[s]['name'] for s in completed)}"
        if remaining:
            stage_guidance += f"\n→ まだ: {', '.join(INTERVIEW_STAGES[s]['name'] for s in remaining[:2])}"
        
        # 問診プランナーからの動的ガイダンスを追加
        planner_guidance = self.interview_planner.get_guidance()
        if planner_guidance:
            stage_guidance += f"\n\n{planner_guidance}"
        
        # 文脈情報を追加
        context_info = self.interview_planner.get_prompt_context()
        if context_info:
            stage_guidance += f"\n\n## 患者との会話の文脈\n{context_info}"
        
        return SYSTEM_PROMPT + stage_guidance

    def _update_stage(self, user_text: str, assistant_response: str):
        """Update interview stage based on conversation content."""
        combined = (user_text + assistant_response).lower()
        
        # Track topics covered
        if any(word in combined for word in ['体調', 'めまい', '喉', '渇き', 'だるい', '調子', '元気', '風邪', '具合']):
            self.stage_completed["health_check"] = True
            self.last_topic = "体調"
        if any(word in combined for word in ['食事', '食べ', '甘い', 'ご飯', '野菜', '間食', '塩分', 'おやつ']):
            self.stage_completed["diet"] = True
            self.last_topic = "食事"
        if any(word in combined for word in ['運動', '散歩', '歩', '外出', '動', '習慣']):
            self.stage_completed["exercise"] = True
            self.last_topic = "運動"
        if any(word in combined for word in ['仕事', '会社', '残業', '忙しい', 'ストレス', '疲れ']):
            self.stage_completed["work"] = True
            self.last_topic = "仕事"
        if any(word in combined for word in ['薬', '服用', '飲み忘れ', '処方', '飲んで', 'インスリン', '注射']):
            self.stage_completed["medication"] = True
            self.last_topic = "薬"
        if any(word in combined for word in ['来院', '次回', '予約', '予定', '病院']):
            self.stage_completed["closing"] = True
            self.last_topic = "次回来院"
        
        # Mark greeting as complete after first exchange
        if len(self.history) >= 2:
            self.stage_completed["greeting"] = True
        
        # Auto-advance to next incomplete stage
        current_info = INTERVIEW_STAGES.get(self.current_stage, {})
        if self.stage_completed.get(self.current_stage, False):
            next_stage = current_info.get("next")
            if next_stage and not self.stage_completed.get(next_stage, False):
                self.current_stage = next_stage
                print(f"Interview stage advanced to: {next_stage}")

    async def process_text(self, text: str) -> AsyncGenerator[tuple[str, str], None]:
        """Process user text and generate response with interview orchestration."""
        text = text.strip()
        
        # Handle short affirmative responses
        if text in ['です', 'はい', 'うん', 'ええ', 'そう', 'ああ', 'うーん']:
            if self.last_topic:
                text = f"（{self.last_topic}について）{text}"
        
        # Combine with pending utterance if any
        if self.pending_utterance:
            text = self.pending_utterance + " " + text
            self.pending_utterance = ""
        
        self.history.append({"role": "user", "content": text})
        
        # 問診プランナーで発話を分析（並列実行想定、レイテンシ増加なし）
        self.interview_planner.analyze_utterance(text)
        
        # Build messages with dynamic system prompt
        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + self.history

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.8,  # Slightly higher for more natural variation
                max_tokens=300   # Shorter for snappier responses
            )

            full_response = response.choices[0].message.content or ""
            
            # Parse JSON
            try:
                parsed = json.loads(full_response)
                display_text = parsed.get("display", "")
                speech_text = parsed.get("speech", display_text)
            except json.JSONDecodeError:
                display_text = full_response
                speech_text = full_response
                print(f"Warning: Failed to parse LLM response as JSON: {full_response[:100]}")
            
            # Update history
            self.history.append({"role": "assistant", "content": display_text})
            
            # Update interview stage
            self._update_stage(text, display_text)
            
            yield (display_text, speech_text)

        except Exception as e:
            print(f"LLM Error: {e}")
            error_msg = "すみません、少し聞き取りにくかったです。もう一度お願いできますか？"
            yield (error_msg, error_msg)

    def get_summary(self) -> dict:
        """Get conversation progress including interview planner data."""
        return {
            "current_stage": self.current_stage,
            "stage_completed": self.stage_completed,
            "history_length": len(self.history),
            "last_topic": self.last_topic,
            # 問診プランナーからのデータを追加
            "interview_analysis": self.interview_planner.get_report_data()
        }

    def reset(self):
        """Reset conversation state."""
        self.history = []
        self.current_stage = "greeting"
        self.stage_completed = {stage: False for stage in INTERVIEW_STAGES}
        self.last_topic = None
        self.pending_utterance = ""
        # 問診プランナーもリセット
        self.interview_planner.reset()
