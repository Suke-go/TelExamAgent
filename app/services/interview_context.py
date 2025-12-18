"""
Interview Context Management Module

低遅延で患者の行動・活動動態を効果的に推論・記録するための文脈管理。
すべての分析はLLM呼び出しと並列で実行し、レイテンシを増加させない。
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import re


# ============================================================
# 列挙型定義
# ============================================================

class EmotionalTone(Enum):
    """患者の感情状態"""
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"      # 不安・心配
    POSITIVE = "positive"    # 元気・調子良い
    TIRED = "tired"          # 疲労・だるい
    FRUSTRATED = "frustrated" # 困惑・イライラ


class BeliefType(Enum):
    """NCF: 患者の信念タイプ"""
    ACCEPTING = "accepting"       # 必要性高・懸念低 → 称賛・維持
    AMBIVALENT = "ambivalent"     # 必要性高・懸念高 → 懸念を傾聴
    INDIFFERENT = "indifferent"   # 必要性低・懸念低 → 必要性の再認識
    SKEPTICAL = "skeptical"       # 必要性低・懸念高 → 傾聴優先
    UNKNOWN = "unknown"           # まだ判定できない


class SeverityLevel(Enum):
    """障壁の深刻度"""
    TRIVIAL = "trivial"      # 軽微：スルーでOK
    MODERATE = "moderate"    # 中程度：確認質問1つ
    SERIOUS = "serious"      # 深刻：詳細に探る


class BarrierCategory(Enum):
    """COM-B障壁カテゴリ"""
    CAPABILITY_PSYCHOLOGICAL = "capability_psychological"
    CAPABILITY_PHYSICAL = "capability_physical"
    OPPORTUNITY_PHYSICAL = "opportunity_physical"
    OPPORTUNITY_SOCIAL = "opportunity_social"
    MOTIVATION_REFLECTIVE = "motivation_reflective"
    MOTIVATION_AUTOMATIC = "motivation_automatic"


# ============================================================
# データクラス
# ============================================================

@dataclass
class IdentifiedBarrier:
    """検出された障壁"""
    category: BarrierCategory
    description: str
    source_utterance: str
    severity: SeverityLevel = SeverityLevel.MODERATE
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "source": self.source_utterance[:100],
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CollectedInfo:
    """問診で収集済みの情報"""
    # 体調関連
    symptoms: List[str] = field(default_factory=list)
    general_condition: Optional[str] = None  # 全体的な体調
    
    # 服薬関連
    medication_adherence: Optional[str] = None  # 飲めている/忘れがち/etc
    medication_barriers: List[str] = field(default_factory=list)
    insulin_used: Optional[bool] = None         # インスリン使用の有無
    medication_regularity: Optional[str] = None  # 規則正しく飲めているか
    
    # 生活習慣
    diet: Dict[str, Any] = field(default_factory=lambda: {
        "frequency": None,    # 回数
        "regularity": None,   # 規則性
        "snacks": None,       # 間食の量
        "salt": None,         # 塩分
        "sugar": None         # 糖分
    })
    
    exercise: Dict[str, Any] = field(default_factory=lambda: {
        "frequency": None,    # 回数
        "habit": None,        # 習慣性
        "content": None       # 内容
    })
    
    work: Dict[str, Any] = field(default_factory=lambda: {
        "overwork": None,     # 過労
        "stress": None        # ストレス
    })
    
    # 旧フィールド（互換性のために残すか、完全に移行するか）
    diet_status: Optional[str] = None
    exercise_status: Optional[str] = None
    
    # 次回来院
    next_visit_confirmed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symptoms": self.symptoms,
            "general_condition": self.general_condition,
            "medication_adherence": self.medication_adherence,
            "medication_barriers": self.medication_barriers,
            "insulin_used": self.insulin_used,
            "medication_regularity": self.medication_regularity,
            "diet": self.diet,
            "exercise": self.exercise,
            "work": self.work,
            "diet_status": self.diet_status,
            "exercise_status": self.exercise_status,
            "next_visit_confirmed": self.next_visit_confirmed
        }


@dataclass
class PatientBeliefProfile:
    """NCF: 患者の信念プロファイル"""
    necessity_signals: List[str] = field(default_factory=list)
    concern_signals: List[str] = field(default_factory=list)
    
    def classify(self) -> BeliefType:
        """信念タイプを分類"""
        necessity_score = len(self.necessity_signals)
        concern_score = len(self.concern_signals)
        
        if necessity_score == 0 and concern_score == 0:
            return BeliefType.UNKNOWN
        
        high_necessity = necessity_score >= 2
        high_concern = concern_score >= 2
        
        if high_necessity and not high_concern:
            return BeliefType.ACCEPTING
        elif high_necessity and high_concern:
            return BeliefType.AMBIVALENT
        elif not high_necessity and not high_concern:
            return BeliefType.INDIFFERENT
        else:
            return BeliefType.SKEPTICAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "necessity_signals": self.necessity_signals,
            "concern_signals": self.concern_signals,
            "belief_type": self.classify().value
        }


@dataclass
class InterviewContext:
    """
    問診の文脈を管理するメインクラス
    
    設計原則:
    - すべての分析はLLM呼び出しと並列で実行（レイテンシ増加なし）
    - 軽量な正規表現・キーワードマッチで即座に抽出
    - 次ターンのLLM呼び出し時に文脈として活用
    """
    # 患者プロファイル
    belief_profile: PatientBeliefProfile = field(default_factory=PatientBeliefProfile)
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    
    # 検出された障壁
    barriers: List[IdentifiedBarrier] = field(default_factory=list)
    
    # 収集済み情報
    collected_info: CollectedInfo = field(default_factory=CollectedInfo)
    
    # 対話状態
    current_topic: Optional[str] = None
    topics_covered: List[str] = field(default_factory=list)
    turn_count: int = 0
    
    # 会話履歴のサマリー（LLMプロンプト用）
    conversation_summary: str = ""
    
    def to_prompt_context(self) -> str:
        """LLMプロンプトに埋め込む文脈サマリーを生成"""
        parts = []
        
        # 信念タイプ
        belief_type = self.belief_profile.classify()
        if belief_type != BeliefType.UNKNOWN:
            parts.append(f"【患者タイプ】{self._belief_type_description(belief_type)}")
        
        # 感情状態
        if self.emotional_tone != EmotionalTone.NEUTRAL:
            parts.append(f"【患者の様子】{self._emotional_tone_description()}")
        
        # 最新の障壁
        if self.barriers:
            latest = self.barriers[-1]
            parts.append(f"【検出された課題】{latest.description}（{latest.severity.value}）")
        
        # 収集済み情報
        info_parts = []
        if self.collected_info.symptoms:
            info_parts.append(f"症状: {', '.join(self.collected_info.symptoms)}")
        if self.collected_info.medication_adherence:
            info_parts.append(f"服薬: {self.collected_info.medication_adherence}")
        if info_parts:
            parts.append(f"【収集済み】{'; '.join(info_parts)}")
        
        # まだ聞いていない話題
        remaining = self._get_remaining_topics()
        if remaining:
            parts.append(f"【未確認】{', '.join(remaining[:3])}")
        
        return "\n".join(parts) if parts else "（初回の会話です）"
    
    def _belief_type_description(self, belief_type: BeliefType) -> str:
        descriptions = {
            BeliefType.ACCEPTING: "服薬の必要性を理解している。称賛して維持。",
            BeliefType.AMBIVALENT: "必要性は感じているが懸念もある。懸念を傾聴。",
            BeliefType.INDIFFERENT: "必要性を低く認識。穏やかに効果を確認。",
            BeliefType.SKEPTICAL: "懸念が強く必要性も低い。まず傾聴優先。",
        }
        return descriptions.get(belief_type, "")
    
    def _emotional_tone_description(self) -> str:
        descriptions = {
            EmotionalTone.ANXIOUS: "不安そう・心配している様子",
            EmotionalTone.POSITIVE: "元気・調子が良さそう",
            EmotionalTone.TIRED: "疲れている・だるそう",
            EmotionalTone.FRUSTRATED: "困っている・イライラ気味",
        }
        return descriptions.get(self.emotional_tone, "")
    
    def _get_remaining_topics(self) -> List[str]:
        all_topics = ["体調", "服薬", "食事", "運動", "次回来院"]
        return [t for t in all_topics if t not in self.topics_covered]
    
    def to_report_dict(self) -> Dict[str, Any]:
        """レポート用のサマリーを生成"""
        return {
            "belief_profile": self.belief_profile.to_dict(),
            "emotional_tone": self.emotional_tone.value,
            "barriers": [b.to_dict() for b in self.barriers],
            "collected_info": self.collected_info.to_dict(),
            "topics_covered": self.topics_covered,
            "turn_count": self.turn_count
        }
