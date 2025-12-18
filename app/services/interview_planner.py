"""
Interview Planner Module

行動科学フレームワークを統合した問診プランナー。
深刻度評価に基づく適応的質問深度を実現。
"""
from typing import Optional, List
from dataclasses import dataclass
from .interview_context import (
    InterviewContext,
    IdentifiedBarrier,
    SeverityLevel,
    BeliefType,
    CollectedInfo
)
from .barrier_detector import (
    BarrierDetector,
    SeverityAssessor,
    BeliefAnalyzer,
    EmotionalToneDetector,
    SymptomExtractor,
    LifestyleExtractor
)


# ============================================================
# 深刻度に応じた応答戦略
# ============================================================

SEVERITY_STRATEGIES = {
    SeverityLevel.TRIVIAL: """
【対応方針】軽く流す
- 「そういうこともありますよね」程度で次の話題へ
- 深堀りしない（過剰な心配は逆効果）
- 例：「たまに忘れることありますよね。でも基本的には飲めているんですね！」
""",
    
    SeverityLevel.MODERATE: """
【対応方針】確認質問1つ
- 頻度や程度を1つ質問して深刻度を確認
- 回答次第で深堀りするか判断
- 例：「飲み忘れがあるんですね。どれくらいの頻度ですか？」
""",
    
    SeverityLevel.SERIOUS: """
【対応方針】詳細に傾聴
- 背景・理由・困っていることを丁寧に聞く
- 介入はせず、傾聴に徹する
- 例：「毎日のように忘れてしまうんですね...何か理由とか、困っていることはありますか？」
""",
}


BELIEF_TYPE_STRATEGIES = {
    BeliefType.ACCEPTING: "患者は服薬の必要性を理解。称賛して次の話題へ。",
    BeliefType.AMBIVALENT: "必要性は感じるが懸念がある。懸念を丁寧に傾聴。",
    BeliefType.INDIFFERENT: "必要性を低く認識。穏やかに効果や重要性を確認。",
    BeliefType.SKEPTICAL: "懸念が強く必要性も低い。まず傾聴を最優先。",
    BeliefType.UNKNOWN: "まだ判断できない。自然に話を聞く。",
}


# ============================================================
# 問診プランナー
# ============================================================

class InterviewPlanner:
    """
    行動科学フレームワークを統合した問診プランナー
    
    設計原則:
    - すべての分析はLLM呼び出しと並列実行（レイテンシ増加なし）
    - 深刻度に応じて質問の深さを適応的に調整
    - 軽微な問題はスルー、深刻な問題のみ詳細に聞く
    """
    
    def __init__(self):
        self.barrier_detector = BarrierDetector()
        self.severity_assessor = SeverityAssessor()
        self.belief_analyzer = BeliefAnalyzer()
        self.emotion_detector = EmotionalToneDetector()
        self.symptom_extractor = SymptomExtractor()
        self.lifestyle_extractor = LifestyleExtractor()
        
        self.context = InterviewContext()
    
    def analyze_utterance(self, user_text: str) -> None:
        """
        ユーザー発話を分析して文脈を更新。
        LLM呼び出しと並列で実行される想定。
        """
        # ターン数を更新
        self.context.turn_count += 1
        
        # 障壁検出
        barriers = self.barrier_detector.detect(user_text)
        for barrier in barriers:
            # 深刻度を評価
            barrier.severity = self.severity_assessor.assess(user_text, barrier)
            self.context.barriers.append(barrier)
        
        # 信念分析
        self.context.belief_profile = self.belief_analyzer.analyze(
            user_text, 
            self.context.belief_profile
        )
        
        # 感情検出
        self.context.emotional_tone = self.emotion_detector.detect(user_text)
        
        # 症状抽出
        symptoms = self.symptom_extractor.extract(user_text)
        for symptom in symptoms:
            if symptom not in self.context.collected_info.symptoms:
                self.context.collected_info.symptoms.append(symptom)
        
        # 生活習慣・治療情報の抽出
        self._update_collected_info(user_text)
        
        # 話題の追跡
        self._track_topics(user_text)

    def _update_collected_info(self, text: str) -> None:
        """詳細情報を抽出して更新"""
        info = self.context.collected_info
        
        # 食事
        diet_details = self.lifestyle_extractor.extract_diet(text)
        for k, v in diet_details.items():
            info.diet[k] = v
            
        # 運動
        exercise_details = self.lifestyle_extractor.extract_exercise(text)
        for k, v in exercise_details.items():
            info.exercise[k] = v
            
        # 仕事
        work_details = self.lifestyle_extractor.extract_work(text)
        for k, v in work_details.items():
            info.work[k] = v
            
        # 服薬・インスリン
        medical_details = self.lifestyle_extractor.extract_medical(text)
        if "regularity" in medical_details:
            info.medication_regularity = medical_details["regularity"]
        if "insulin_used" in medical_details:
            info.insulin_used = medical_details["insulin_used"]
    
    def _track_topics(self, text: str) -> None:
        """話題の追跡"""
        topic_patterns = {
            "体調": r"(体調|調子|具合|元気)",
            "服薬": r"(薬|くすり|飲んで|服用|インスリン|注射)",
            "食事": r"(食事|食べ|ご飯|野菜|甘い|間食|塩分)",
            "運動": r"(運動|散歩|歩|動)",
            "仕事": r"(仕事|会社|残業|ストレス)",
            "次回来院": r"(来院|病院|予約|次回)",
        }
        import re
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, text) and topic not in self.context.topics_covered:
                self.context.current_topic = topic
                self.context.topics_covered.append(topic)
    
    def get_guidance(self) -> str:
        """
        LLMに渡すガイダンスを生成。
        """
        parts = []
        
        # 1. 患者の信念タイプに基づくアドバイス
        belief_type = self.context.belief_profile.classify()
        if belief_type != BeliefType.UNKNOWN:
            parts.append(f"## 患者タイプ\n{BELIEF_TYPE_STRATEGIES[belief_type]}")
        
        # 2. 最新の障壁への対応
        if self.context.barriers:
            latest = self.context.barriers[-1]
            severity_strategy = SEVERITY_STRATEGIES[latest.severity]
            
            parts.append(f"## 検出された課題: {latest.description}")
            parts.append(severity_strategy)
            
            # 軽微な場合は明示的に「流して」と指示
            if latest.severity == SeverityLevel.TRIVIAL:
                parts.append("→ この話題は軽く流して、次へ進んでください。")
        
        # 3. 未確認の聞き取り項目への誘導
        info = self.context.collected_info
        missing = []
        
        # 食事の深掘りが必要か
        if self.context.current_topic == "食事":
            if not info.diet["frequency"]: missing.append("1日の食事回数")
            if not info.diet["regularity"]: missing.append("食事の規則性")
            if not info.diet["snacks"]: missing.append("間食")
            if not info.diet["salt"]: missing.append("塩分量")
            if not info.diet["sugar"]: missing.append("糖分・甘いもの")
            
        # 運動の深掘り
        if self.context.current_topic == "運動":
            if not info.exercise["habit"]: missing.append("運動の習慣性")
            if not info.exercise["content"]: missing.append("具体的な内容")

        # 仕事の深掘り
        if self.context.current_topic == "仕事":
            if not info.work["overwork"]: missing.append("過労の有無")
            if not info.work["stress"]: missing.append("ストレス")

        if missing:
            parts.append(f"## 追加で確認したい点（自然に）\n{', '.join(missing)}")
        
        # 4. 次に聞くべき話題
        remaining = self._get_remaining_topics()
        if remaining and not self.context.barriers and not missing:
            parts.append(f"## 次の話題候補\n{', '.join(remaining[:2])}")
        
        return "\n\n".join(parts) if parts else ""
    
    def _get_remaining_topics(self) -> List[str]:
        """まだカバーしていない話題を取得"""
        all_topics = ["体調", "食事", "運動", "仕事", "服薬", "次回来院"]
        return [t for t in all_topics if t not in self.context.topics_covered]
    
    def get_prompt_context(self) -> str:
        """LLMプロンプトに埋め込む文脈情報"""
        return self.context.to_prompt_context()
    
    def get_report_data(self) -> dict:
        """レポート用のデータを取得"""
        return self.context.to_report_dict()
    
    def reset(self) -> None:
        """文脈をリセット"""
        self.context = InterviewContext()
