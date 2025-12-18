
import sys
import os
import asyncio

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.interview_planner import InterviewPlanner

async def test_extraction():
    planner = InterviewPlanner()
    
    test_cases = [
        {
            "text": "最近、食事は1日3回規則正しく食べています。でもついつい甘い間食を摂りすぎちゃうんですよね。塩分も少し気にして薄味にしています。",
            "topic": "食事",
            "expected_info": {
                "diet": {
                    "regularity": "確認済み",
                    "snacks": "言及あり",
                    "salt": "言及あり",
                    "sugar": "言及あり"
                }
            }
        },
        {
            "text": "散歩は毎日30分くらい歩くようにしています。ラジオ体操もたまにやりますよ。",
            "topic": "運動",
            "expected_info": {
                "exercise": {
                    "habit": "習慣化",
                    "frequency": "言及あり",
                    "content": "言及あり"
                }
            }
        },
        {
            "text": "最近は仕事が忙しくて残業続きで、結構ストレスが溜まっている気がします。",
            "topic": "仕事",
            "expected_info": {
                "work": {
                    "overwork": "言及あり",
                    "stress": "言及あり"
                }
            }
        },
        {
            "text": "薬は毎日ちゃんと飲んでいます。インスリンの注射も忘れずに打っていますよ。",
            "topic": "服薬",
            "expected_info": {
                "medication_regularity": "良好",
                "insulin_used": True
            }
        },
        # 新規テストケース: 食事回数の検出
        {
            "text": "1日2食しか食べられてないです。朝は時間がなくて抜いています。",
            "topic": "食事",
            "expected_info": {
                "diet": {
                    "frequency": "言及あり",
                }
            }
        },
        # 新規テストケース: 服薬「要注意」パターン
        {
            "text": "薬はたまに飲み忘れちゃうこともあるんですよね。",
            "topic": "服薬",
            "expected_info": {
                "medication_regularity": "要注意",
            }
        },
    ]
    
    for case in test_cases:
        print(f"\n--- Testing: {case['text']} ---")
        planner.analyze_utterance(case['text'])
        info = planner.context.collected_info
        
        # Verify extraction
        for category, expected in case["expected_info"].items():
            actual = getattr(info, category)
            if isinstance(actual, dict):
                for k, v in expected.items():
                    if actual.get(k) == v:
                        print(f"[OK] {category}.{k} = {v}")
                    else:
                        print(f"[FAIL] {category}.{k} expected {v}, got {actual.get(k)}")
            else:
                if actual == expected:
                    print(f"[OK] {category} = {expected}")
                else:
                    print(f"[FAIL] {category} expected {expected}, got {actual}")

        if planner.context.current_topic == case["topic"]:
            print(f"[OK] Topic matched: {case['topic']}")
        else:
            print(f"[FAIL] Topic mismatch: expected {case['topic']}, got {planner.context.current_topic}")

if __name__ == "__main__":
    asyncio.run(test_extraction())
