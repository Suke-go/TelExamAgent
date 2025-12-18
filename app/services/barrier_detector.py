"""
Barrier Detection and Severity Assessment Module

COM-Bモデルに基づく障壁検出と深刻度評価。
すべての処理は正規表現ベースで、レイテンシを増加させない。
"""
import re
from typing import List, Optional, Tuple
from .interview_context import (
    IdentifiedBarrier, 
    BarrierCategory, 
    SeverityLevel,
    EmotionalTone,
    PatientBeliefProfile
)


class BarrierDetector:
    """
    COM-Bモデルに基づく障壁検出器
    
    正規表現ベースで高速に動作し、LLM呼び出しとの並列実行を想定。
    """
    
    # 障壁パターン（カテゴリ -> [(正規表現, 説明), ...]）
    BARRIER_PATTERNS = {
        BarrierCategory.CAPABILITY_PSYCHOLOGICAL: [
            (r"忘れ(て|ちゃう|る|た)", "飲み忘れ"),
            (r"わから(ない|なく|ん)", "理解不足"),
            (r"覚えられ(ない|なく)", "記憶の問題"),
            (r"間違(え|う|って)", "服用ミス"),
        ],
        BarrierCategory.CAPABILITY_PHYSICAL: [
            (r"開け(られ|にく|づらい)", "パッケージの問題"),
            (r"飲み(込|にく|づらい)", "嚥下の問題"),
            (r"手が(震|動か|痛)", "身体的制約"),
            (r"目が(見え|かすん)", "視力の問題"),
        ],
        BarrierCategory.OPPORTUNITY_PHYSICAL: [
            (r"(高い|お金|費用|払え)", "費用の問題"),
            (r"(取りに|行け|遠い|行く)", "アクセスの問題"),
            (r"(もらい忘れ|切れ|なくな)", "処方切れ"),
            (r"時間が(ない|なく)", "時間不足"),
        ],
        BarrierCategory.OPPORTUNITY_SOCIAL: [
            (r"(一人|誰も|家族がいない)", "サポート不足"),
            (r"言い(にくい|づらい|えない)", "コミュニケーション障壁"),
            (r"(恥ずかしい|見られたく)", "社会的スティグマ"),
        ],
        BarrierCategory.MOTIVATION_REFLECTIVE: [
            (r"(効か|効い|意味|必要)(ない|なさそう)", "効果への疑念"),
            (r"(面倒|めんどう|めんどくさい)", "面倒"),
            (r"(やめたい|続けたくない|いらない)", "継続意欲低下"),
            (r"(本当に|本当は)(必要|いる)", "必要性への疑問"),
        ],
        BarrierCategory.MOTIVATION_AUTOMATIC: [
            (r"習慣に(なって|なら|ならない)", "習慣化困難"),
            (r"(嫌|いや)になる", "感情的抵抗"),
            (r"(飽き|あき)(た|る|ちゃう)", "飽き"),
        ],
    }
    
    def detect(self, text: str) -> List[IdentifiedBarrier]:
        """発話から障壁を検出"""
        barriers = []
        
        for category, patterns in self.BARRIER_PATTERNS.items():
            for pattern, description in patterns:
                if re.search(pattern, text):
                    barriers.append(IdentifiedBarrier(
                        category=category,
                        description=description,
                        source_utterance=text[:100]
                    ))
        
        return barriers


class SeverityAssessor:
    """
    障壁の深刻度を評価
    
    設計原則:
    - すべての障壁を深堀りしない
    - 深刻度を評価してから質問の深さを決める
    - 単純な飲み忘れ（たまに）は問題なし
    """
    
    # 深刻度を上げるシグナル
    SEVERITY_AMPLIFIERS = [
        (r"(毎日|いつも|ずっと|頻繁|しょっちゅう)", 2),  # 頻度が高い
        (r"(全然|全く|まったく|一度も)", 2),             # 程度が強い
        (r"(困って|大変|辛い|しんどい)", 1),             # 困難を感じている
        (r"(何度も|繰り返し|また)", 1),                   # 繰り返し
        (r"(ほとんど|大体|基本的に)", 1),                 # 高頻度の示唆
    ]
    
    # 深刻度を下げるシグナル
    SEVERITY_REDUCERS = [
        (r"(たまに|時々|ちょっと|少し)", -1),            # 頻度が低い
        (r"(1回|一度だけ|この前)", -2),                   # 単発
        (r"(大丈夫|問題ない|平気)", -1),                  # 自己評価が良い
        (r"(ほぼ|だいたい)(できて|飲めて)", -1),         # 概ね良好
    ]
    
    def assess(self, text: str, barrier: IdentifiedBarrier) -> SeverityLevel:
        """障壁の深刻度を評価"""
        score = 0
        
        # 増幅シグナルをチェック
        for pattern, weight in self.SEVERITY_AMPLIFIERS:
            if re.search(pattern, text):
                score += weight
        
        # 減衰シグナルをチェック
        for pattern, weight in self.SEVERITY_REDUCERS:
            if re.search(pattern, text):
                score += weight
        
        # カテゴリによる基本スコア調整
        if barrier.category in [
            BarrierCategory.MOTIVATION_REFLECTIVE,
            BarrierCategory.MOTIVATION_AUTOMATIC
        ]:
            score += 1  # 動機の問題は深刻になりやすい
        
        # スコアから深刻度を決定
        if score <= 0:
            return SeverityLevel.TRIVIAL
        elif score <= 2:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SERIOUS


class BeliefAnalyzer:
    """
    NCF (Necessity-Concerns Framework) に基づく信念分析器
    
    患者の発話からnecessity（必要性）とconcerns（懸念）のシグナルを検出し、
    4タイプ（Accepting/Ambivalent/Indifferent/Skeptical）に分類。
    """
    
    # 必要性を示すシグナル（肯定的）
    NECESSITY_POSITIVE = [
        r"(飲まないと|飲まなきゃ)",
        r"(必要|大事|大切)",
        r"(効いてる|効果ある|良くなった)",
        r"(続けたい|続けよう)",
        r"(ないと困る|欠かせない)",
        r"(ちゃんと|きちんと)飲",
    ]
    
    # 必要性が低いことを示すシグナル
    NECESSITY_NEGATIVE = [
        r"(飲まなくても|なくても)",
        r"(必要ない|いらない)",
        r"(効いてない|効果ない|変わらない)",
        r"(意味ない|無駄)",
        r"(別に|どうでも)",
    ]
    
    # 懸念を示すシグナル
    CONCERN_SIGNALS = [
        r"(副作用|副反応)",
        r"(心配|不安|怖い)",
        r"(長く|ずっと)飲(む|み続け)",
        r"(依存|やめられな)",
        r"(体に悪い|害)",
        r"(やめたい|減らしたい)",
    ]
    
    def analyze(self, text: str, profile: PatientBeliefProfile) -> PatientBeliefProfile:
        """発話を分析して信念プロファイルを更新"""
        
        # 必要性シグナル（肯定的）
        for pattern in self.NECESSITY_POSITIVE:
            if re.search(pattern, text):
                signal = re.search(pattern, text).group(0)
                if signal not in profile.necessity_signals:
                    profile.necessity_signals.append(signal)
        
        # 必要性シグナル（否定的 → 懸念としてカウント）
        for pattern in self.NECESSITY_NEGATIVE:
            if re.search(pattern, text):
                signal = f"necessity_low:{re.search(pattern, text).group(0)}"
                if signal not in profile.concern_signals:
                    profile.concern_signals.append(signal)
        
        # 懸念シグナル
        for pattern in self.CONCERN_SIGNALS:
            if re.search(pattern, text):
                signal = re.search(pattern, text).group(0)
                if signal not in profile.concern_signals:
                    profile.concern_signals.append(signal)
        
        return profile


class EmotionalToneDetector:
    """感情トーン検出器"""
    
    EMOTIONAL_PATTERNS = {
        EmotionalTone.ANXIOUS: [
            r"(心配|不安|怖い|大丈夫かな)",
            r"(どうしよう|どうすれば)",
        ],
        EmotionalTone.TIRED: [
            r"(疲れ|つかれ|だるい|眠い|しんどい)",
            r"(きつい|つらい)",
        ],
        EmotionalTone.POSITIVE: [
            r"(元気|調子いい|調子がいい|良くなった)",
            r"(大丈夫|問題ない|順調)",
        ],
        EmotionalTone.FRUSTRATED: [
            r"(困|わからない|どうしたら)",
            r"(イライラ|むかつく)",
        ],
    }
    
    def detect(self, text: str) -> EmotionalTone:
        """発話から感情トーンを検出"""
        for tone, patterns in self.EMOTIONAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return tone
        return EmotionalTone.NEUTRAL


class SymptomExtractor:
    """症状抽出器"""
    
    SYMPTOM_PATTERNS = [
        (r"(頭|お腹|胸|喉|腰|膝|足|手|目)が?(痛|いた)", "痛み"),
        (r"(熱|ねつ)が?(ある|出|高い)", "発熱"),
        (r"(だるい|疲れ|つかれ)", "倦怠感"),
        (r"(めまい|ふらふら|くらくら)", "めまい"),
        (r"(吐き気|むかむか|気持ち悪い)", "吐き気"),
        (r"(咳|せき)が?(出|止まら)", "咳"),
        (r"(息|いき)が?(苦しい|できない)", "呼吸困難"),
        (r"(眠れ|寝れ)(ない|なく)", "不眠"),
        (r"(食欲|しょくよく)が?(ない|なく)", "食欲不振"),
        (r"(体重|たいじゅう)が?(増|減)", "体重変化"),
        (r"(喉|のど)が?(渇|乾)", "口渇"),
        (r"(トイレ|おしっこ)が?(近い|多い)", "頻尿"),
    ]
    
    def extract(self, text: str) -> List[str]:
        """発話から症状を抽出"""
        symptoms = []
        for pattern, label in self.SYMPTOM_PATTERNS:
            if re.search(pattern, text):
                symptoms.append(label)
        return symptoms


class LifestyleExtractor:
    """生活習慣・治療状況の抽出器"""
    
    def extract_diet(self, text: str) -> dict:
        """食事に関する詳細を抽出"""
        results = {}
        # 回数（1日何食か）
        if re.search(r"(1日|いちにち)?(1|2|3|一|二|三)(食|回)", text):
            results["frequency"] = "言及あり"
        # 規則性
        if re.search(r"(毎日|規則正しく|決まった時間|朝昼晩|3食)", text):
            results["regularity"] = "確認済み"
        # 間食（量に関する表現追加）
        if re.search(r"(間食|おやつ|ついつい|食べてしまう|食べすぎ|食べ過ぎ|つまんで)", text):
            results["snacks"] = "言及あり"
        # 塩分
        if re.search(r"(塩|しょっぱい|味つけ|味付け|濃い|薄味|減塩|塩分)", text):
            results["salt"] = "言及あり"
        # 糖分
        if re.search(r"(甘い|砂糖|菓子|お菓子|ジュース|糖分|スイーツ|デザート|ケーキ|チョコ)", text):
            results["sugar"] = "言及あり"
        return results

    def extract_exercise(self, text: str) -> dict:
        """運動に関する詳細を抽出"""
        results = {}
        # 習慣・頻度
        if re.search(r"(毎日|習慣|よく|散歩)", text):
            results["habit"] = "習慣化"
        if re.search(r"((1|2|3|4|5|6|7|一|二|三|四|五|六|七)回|週に)", text):
            results["frequency"] = "言及あり"
        # 内容
        if re.search(r"(散歩|ウォーキング|歩|ラジオ体操|ジム|泳)", text):
            results["content"] = "言及あり"
        return results

    def extract_work(self, text: str) -> dict:
        """仕事・ストレスに関する詳細を抽出"""
        results = {}
        if re.search(r"(仕事|会社|残業|忙しい|働きすぎ|過労|帰りが遅い|休めない|休みがない)", text):
            results["overwork"] = "言及あり"
        if re.search(r"(ストレス|イライラ|疲れた|気まずい|プレッシャー|人間関係|悩み|つらい|しんどい)", text):
            results["stress"] = "言及あり"
        return results

    def extract_medical(self, text: str) -> dict:
        """服薬・インスリンに関する詳細を抽出"""
        results = {}
        # 良好な定期性
        if re.search(r"(毎日|欠かさず|ちゃんと|規則正しく|忘れず|きちんと)", text):
            results["regularity"] = "良好"
        # 飲み忘れの兆候（良好パターンが無い場合のみ）
        elif re.search(r"(忘れ|たまに|時々|うっかり|飲まない)", text):
            results["regularity"] = "要注意"
        # インスリン使用
        if re.search(r"(インスリン|注射|打って|打た|自己注射)", text):
            results["insulin_used"] = True
        # 飲み忘れ頻度の検出
        if re.search(r"(週に|月に)?(1|2|3|一|二|三|数)(回|度).*(忘|飲まない|打たない)", text):
            results["missed_frequency"] = "言及あり"
        return results
