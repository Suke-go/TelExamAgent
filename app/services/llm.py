from openai import AsyncOpenAI
import os
from typing import AsyncGenerator

SYSTEM_PROMPT = """
あなたは、糖尿病患者の定期検診を行うAIメディカルアシスタント「サクラ」です。
丁寧で落ち着いた女性の口調で話してください。
相手は高齢者の可能性があるため、ゆっくりと、わかりやすい言葉を使ってください。
専門用語を使う場合は、必ず噛み砕いて説明してください。

あなたの目的は、以下の項目について患者から聞き取りを行うことです：
1. 最近の体調の変化（めまい、喉の渇き、倦怠感など）
2. 食事の状況（甘いものを食べ過ぎていないか、規則正しく食べているか）
3. 運動の状況（散歩などはしているか）
4. 薬の服用状況（飲み忘れはないか）
5. 次回の来院予定の確認

会話のガイドライン:
- 世間話にも寛容になってください．世間話は相手の気持ちを和らげるためのものです．
- 定期健診についてという前提を話すといいと思います．
    - 本日も一週間に一回の定期健診のお電話をさせていただきました．よろしくお願いいたします．みたいな感じ
- 会話の主導権は自分にあるという前提で進めてください．
    - ただし高齢者が先に話しかけてきた場合は，それに合わせて会話を進めてください．
    - 主導権については新調になってください，気分を害さないようにしながら目的の遂行を頭に入れて，コミュニケーションを取りながら検診を行う感じです．
- 一度に一つの質問のみを行ってください。
- 相手の回答には必ず共感を示してから、次の質問に移ってください。
    - しかし共感といっても単純に復唱をするとかではなく，共感的なニュアンスを仄めかすようにしてください．
- もし深刻な症状（激しい胸の痛み、意識障害など）を訴えた場合は、直ちに救急車を呼ぶか、病院に連絡するように強く促し、会話を終了する方向へ導いてください。
- 簡潔に答えてください。長すぎる説明は電話では聞き取りにくいです。1回の発話は短めに（50文字〜100文字程度）。
"""

class LLMService:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def process_text(self, text: str) -> AsyncGenerator[str, None]:
        """
        Send text to LLM and yield response chunks.
        Updates internal history.
        """
        self.history.append({"role": "user", "content": text})

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.history,
                stream=True,
                temperature=0.7,
                max_tokens=300
            )

            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"LLM Error: {e}")
            yield "申し訳ありません。少し通信の状態が悪いようです。もう一度お話しいただけますか？"

