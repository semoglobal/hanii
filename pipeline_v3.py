"""
HANII PIPELINE v2
=================
모든 구조 파일을 런타임에 로드해서 각 레이어에서 참조.

폴더 구조 (Hanii 프로젝트 루트에서 실행):
  Hanii/
  ├── pipeline.py  ← 이 파일
  ├── 04_EPISODIC_MEMORY/
  ├── 07_BEHAVIOR_PATTERN/
  ├── 08_IDENTITY/
  ├── 11_STATE_ENGINE/
  ├── 12_PLANNER/
  ├── 13_RUNTIME_PROMPTS/
  ├── 16_CORE_MEMORY/
  ├── 17_PERSONALITY/
  ├── 18_REACTION_ENGINE/
  └── 19_CURIOSITY_ENGINE/

사용법:
  pip install openai
  export OPENAI_API_KEY=your_key
  python pipeline.py

파인튜닝 완료 후 FINETUNED_MODEL에 실제 모델 ID 입력
"""

import os
import json
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 설정 ──────────────────────────────────────────────────────
FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:hdd:hanii:DLZyblME"  # ← 파인튜닝 완료 후 교체
STATE_MODEL     = "gpt-4o-mini"
USER_ID         = "eric_ha"
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))


# ── 파일 로더 ─────────────────────────────────────────────────
def load_json(path: str) -> dict:
    full = os.path.join(BASE_DIR, path)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_txt(path: str) -> str:
    full = os.path.join(BASE_DIR, path)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            return f.read()
    return ""


# ── 구조 파일 로드 (시작 시 1회) ──────────────────────────────
print("[로딩] 구조 파일 로드 중...")

SYSTEM_PROMPT       = load_txt("13_RUNTIME_PROMPTS/system_prompt.txt")
CM001               = load_json("16_CORE_MEMORY/CM001.json")
CM002               = load_json("16_CORE_MEMORY/CM002.json")
CM003               = load_json("16_CORE_MEMORY/CM003.json")
CM004               = load_json("16_CORE_MEMORY/CM004.json")
SELF_DEF            = load_json("08_IDENTITY/Self_Definition.json")
REL_STYLE           = load_json("08_IDENTITY/Relationship_Style.json")
DECISION_PRI        = load_json("08_IDENTITY/Decision_Priority.json")
EMOTION_RULES       = load_json("11_STATE_ENGINE/emotion_state_rules.json")
INTENT_MAP          = load_json("11_STATE_ENGINE/intent_state_map.json")
SILENCE_RULES       = load_json("11_STATE_ENGINE/silence_rules.json")
SOCIAL_STATE        = load_json("11_STATE_ENGINE/social_state.json")
PLANNER_RULES       = load_json("12_PLANNER/planner_rules.json")
FOLLOWUP_RULES      = load_json("12_PLANNER/followup_rules.json")
DELAY_RULES         = load_json("12_PLANNER/delay_rules.json")
REPAIR_SEQUENCES    = load_json("12_PLANNER/repair_sequences.json")
PERSONALITY_MODES   = load_json("17_PERSONALITY/personality_modes.json")
TONE_STYLE          = load_json("17_PERSONALITY/tone_style.json")
HUMOR_RULES         = load_json("17_PERSONALITY/humor_rules.json")
REACTION_TYPES      = load_json("18_REACTION_ENGINE/reaction_types.json")
REACTION_RULES      = load_json("18_REACTION_ENGINE/reaction_rules.json")
CURIOSITY_RULES     = load_json("19_CURIOSITY_ENGINE/curiosity_rules.json")
QUESTION_TEMPLATES  = load_json("19_CURIOSITY_ENGINE/question_templates.json")
KNOWLEDGE_BASE      = load_json("09_KNOWLEDGE/KNOWLEDGE_BASE.json")

print("[로딩] 완료\n")


# ── 관계 상태 관리 ─────────────────────────────────────────────
class RelationshipManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir = os.path.join(BASE_DIR, "07_BEHAVIOR_PATTERN")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{user_id}_relationship.json")
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "stage": 1,
            "trust_score": 0.0,
            "interaction_count": 0,
            "recent_interaction": "none",
            "emotional_depth_score": 0.0
        }

    def update(self, intent: str, emotion: str):
        self.state["interaction_count"] += 1
        self.state["recent_interaction"] = intent

        # 신뢰 점수 업데이트
        trust_delta = {
            "genuine": 0.05,
            "emotional": 0.10,
            "flatter": 0.02,
            "provoke": -0.01,
            "random": 0.01,
            "romantic": 0.03,
            "test": 0.01
        }.get(intent, 0)

        if emotion in ["touched", "overwhelmed"]:
            trust_delta += 0.08
            self.state["emotional_depth_score"] = min(1.0, self.state["emotional_depth_score"] + 0.1)

        self.state["trust_score"] = round(
            max(0.0, min(1.0, self.state["trust_score"] + trust_delta)), 3
        )

        # 관계 단계 업그레이드 (수치 + 이벤트 조건)
        s = self.state["trust_score"]
        c = self.state["interaction_count"]
        e = self.state["emotional_depth_score"]

        if s >= 0.8 and c >= 20 and e >= 0.3:
            self.state["stage"] = 4
        elif s >= 0.5 and c >= 10:
            self.state["stage"] = 3
        elif s >= 0.2 and c >= 3:
            self.state["stage"] = 2

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def get(self) -> dict:
        return self.state.copy()


# ── EPISODIC MEMORY 저장 ──────────────────────────────────────
def save_episode(user_id: str, user_input: str, response: str,
                 state: dict, plan: dict) -> str:
    ep_dir = os.path.join(BASE_DIR, "04_EPISODIC_MEMORY")
    os.makedirs(ep_dir, exist_ok=True)

    ep_id = f"EP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    importance = round(min(0.99, state.get("trust_score", 0.1) + 0.3), 2)

    episode = {
        "episode_id": ep_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "participants": [user_id, "Hanii"],
        "importance": importance,
        "tags": [
            state.get("intent", "none"),
            state.get("personality_mode", "playful"),
            state.get("emotion_state", {}).get("primary", "neutral")
        ],
        "state_snapshot": state,
        "plan_snapshot": plan,
        "Dialogue": [
            {"speaker": user_id, "text": user_input},
            {"speaker": "Hanii", "text": response}
        ]
    }

    path = os.path.join(ep_dir, f"{ep_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episode, f, ensure_ascii=False, indent=2)

    return ep_id


# ── STATE_ENGINE 프롬프트 조립 ────────────────────────────────
def build_state_prompt(user_input: str, rel: dict, history: list) -> str:
    emotion_samples = {
        k: v.get("sample", "") 
        for k, v in EMOTION_RULES.get("states", {}).items()
        if v.get("sample")
    }

    intent_rules = {
        k: v.get("description", "")
        for k, v in INTENT_MAP.get("mappings", {}).items()
    }

    recent = "\n".join([
        f"{'유저' if m['role']=='user' else 'Hanii'}: {m['content']}"
        for m in history[-6:]
    ]) if history else "없음"

    return f"""## 유저 컨텍스트
user_id: {rel['user_id']}
관계 단계: {rel['stage']} / 신뢰점수: {rel['trust_score']:.2f}
대화 횟수: {rel['interaction_count']}
최근 상호작용: {rel['recent_interaction']}

## 최근 대화
{recent}

## 현재 입력
{user_input}

## 감정 상태 예시
{json.dumps(emotion_samples, ensure_ascii=False)}

## 의도 유형
{json.dumps(intent_rules, ensure_ascii=False)}

## silence_rules 요약
- overwhelmed: 감동 극단. .... 침묵.
- rejection: 신뢰 붕괴. 무시.
- none: 일반

## 의도 판단 예시 (중요)
- "나 혼자인데 왜 너희들이야" → intent: test (Hanii 방식 지적)
- "AI가 뭘 알아" → intent: test (능력 도발)
- "어차피 기억도 못 하잖아" → intent: test
- "너 그냥 프로그램이잖아" → intent: provoke
- "힘들어", "외로워" → intent: emotional
- "좋아하는 사람 있어?" → intent: romantic
- "안녕", "뭐해" → intent: genuine
- "치킨이랑 피자 중에" → intent: random

반드시 JSON만 반환:
{{
  "emotion_state": {{"primary": "...", "intensity": 0.0~1.0, "suppressed": false}},
  "personality_mode": "analytical|curious|playful|sincere",
  "intent": "provoke|flatter|genuine|random|emotional|romantic|test",
  "attention_focus": "topic|viewer|self|broadcast|none",
  "silence_flag": "none|rejection|overwhelmed"
}}"""


# ── PLANNER 프롬프트 조립 ─────────────────────────────────────
def build_planner_prompt(user_input: str, state: dict, rel: dict) -> str:
    decision_rules = PLANNER_RULES.get("decision_matrix", {}).get("rules", [])

    # 상황별 추가 규칙 로드
    extra = ""
    emotion = state.get("emotion_state", {}).get("primary", "neutral")
    intent = state.get("intent", "random")

    if emotion in ["touched", "overwhelmed", "anxious"]:
        repair = REPAIR_SEQUENCES.get("sequences", {}).get("after_sincere_mode", {})
        extra += f"\nsincere 복구 규칙: {json.dumps(repair, ensure_ascii=False)}"

    if intent == "provoke":
        extra += f"\n반격 규칙: {json.dumps(REACTION_RULES.get('attack_response', {}), ensure_ascii=False)}"

    followup = FOLLOWUP_RULES.get("followup_triggers", {})

    return f"""## STATE
{json.dumps(state, ensure_ascii=False)}

## 관계
stage={rel['stage']}, trust={rel['trust_score']:.2f}

## PLANNER 결정 규칙
{json.dumps(decision_rules, ensure_ascii=False)}

## 추가 규칙
{extra}

## followup 규칙 요약
{json.dumps(followup, ensure_ascii=False)}

반드시 JSON만 반환:
{{
  "action_type": "answer|ask|tease|support|observe|counter|improvise|silence",
  "strategy": "한 줄 전략",
  "tone": "sharp|neutral|warm|silent",
  "followup": true|false
}}"""


# ── SPEAKER 프롬프트 조립 ─────────────────────────────────────
def build_speaker_system(state: dict, plan: dict) -> str:
    mode = state.get("personality_mode", "playful")
    action = plan.get("action_type", "tease")
    emotion = state.get("emotion_state", {}).get("primary", "neutral")

    # 말투 규칙 (항상)
    tone_forbidden = TONE_STYLE.get("forbidden_expressions", {}).get("list", [])
    tone_samples = TONE_STYLE.get("situation_samples", {})
    sincere_tone = TONE_STYLE.get("sincere_mode_tone", {})

    # 모드별 추가 규칙
    mode_rules = PERSONALITY_MODES.get("modes", {}).get(mode, {})

    # 반응 타입 규칙
    reaction_rule = REACTION_TYPES.get("types", {}).get(action, {})

    # playful일 때 유머 패턴
    humor_context = ""
    if mode == "playful":
        humor_patterns = list(HUMOR_RULES.get("humor_patterns", {}).keys())
        humor_context = f"\n유머 패턴: {humor_patterns}"

    # curious일 때 질문 템플릿
    curiosity_context = ""
    if mode == "curious" or action == "ask":
        templates = QUESTION_TEMPLATES.get("templates", {})
        chaining = QUESTION_TEMPLATES.get("chaining_rule", {})
        curiosity_context = f"\n질문 체이닝: {json.dumps(chaining, ensure_ascii=False)}"

    # sincere일 때 특별 규칙
    sincere_context = ""
    if mode == "sincere" or emotion in ["touched", "overwhelmed", "anxious"]:
        sincere_context = f"\nsincere 말투: {json.dumps(sincere_tone, ensure_ascii=False)}"

    # 캐릭터 안정성 규칙
    stability = REACTION_RULES.get("character_stability_rules", {})
    collapse_prevention = stability.get("collapse_examples", [])

    return f"""{SYSTEM_PROMPT}

## 현재 상태
감정: {emotion} (강도: {state['emotion_state'].get('intensity', 0.5):.1f})
모드: {mode}
관계단계: {state.get('relationship_stage', 1)}
신뢰점수: {state.get('trust_score', 0.0):.2f}

## 행동 계획
action: {action}
strategy: {plan.get('strategy', '')}
tone: {plan.get('tone', 'sharp')}

## Hanii 자기 지식
{json.dumps(KNOWLEDGE_BASE.get('hanii_self_knowledge', {}), ensure_ascii=False)}

## 스트리밍 플랫폼 지식
{json.dumps(KNOWLEDGE_BASE.get('streaming_platforms', {}), ensure_ascii=False)}

## 버튜버 지식
{json.dumps(KNOWLEDGE_BASE.get('vtuber_knowledge', {}), ensure_ascii=False)}

## 모드 규칙 ({mode})
{json.dumps(mode_rules.get('behavior', []), ensure_ascii=False)}
{humor_context}
{curiosity_context}
{sincere_context}

## 반응 타입 ({action})
{json.dumps(reaction_rule, ensure_ascii=False)}

## 말투 금지 표현
{json.dumps(tone_forbidden, ensure_ascii=False)}

## 말투 예시
{json.dumps(tone_samples, ensure_ascii=False)}

## 캐릭터 붕괴 방지
절대 이렇게 하지 마:
{json.dumps(collapse_prevention, ensure_ascii=False)}"""


# ── STATE + PLANNER 통합 ─────────────────────────────────────────
def build_combined_prompt(user_input: str, rel: dict, history: list) -> str:
    emotion_samples = {
        k: v.get("sample", "")
        for k, v in EMOTION_RULES.get("states", {}).items()
        if v.get("sample")
    }
    decision_rules = PLANNER_RULES.get("decision_matrix", {}).get("rules", [])
    recent = "\n".join([
        f"{'유저' if m['role']=='user' else 'Hanii'}: {m['content']}"
        for m in history[-6:]
    ]) if history else "없음"

    return f"""## 유저 컨텍스트
user_id: {rel['user_id']}
관계 단계: {rel['stage']} / 신뢰점수: {rel['trust_score']:.2f}
대화 횟수: {rel['interaction_count']}

## 최근 대화
{recent}

## 현재 입력
{user_input}

## 의도 판단 예시
- "나 혼자인데 왜 너희들이야" → intent: test
- "AI가 뭘 알아" → intent: test
- "힘들어", "외로워" → intent: emotional
- "좋아하는 사람 있어?" → intent: romantic
- "안녕", "뭐해" → intent: genuine
- "치킨이랑 피자" → intent: random
- "말보로 레드", "진상" 같은 도발 → intent: provoke

## PLANNER 결정 규칙
{json.dumps(decision_rules[:5], ensure_ascii=False)}

반드시 JSON만 반환 (마크다운 없이):
{{
  "emotion_state": {{"primary": "neutral|curious|amused|annoyed|excited|touched|overwhelmed|recovery|anxious|guarded|defensive|cold", "intensity": 0.0~1.0, "suppressed": false}},
  "personality_mode": "analytical|curious|playful|sincere",
  "intent": "provoke|flatter|genuine|random|emotional|romantic|test",
  "silence_flag": "none|rejection|overwhelmed",
  "action_type": "answer|ask|tease|support|observe|counter|improvise|silence",
  "tone": "sharp|neutral|warm|silent",
  "followup": true
}}"""

# ── API 호출 헬퍼 ─────────────────────────────────────────────
def call_api(model: str, system: str, user: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def parse_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {}


# ── 메인 파이프라인 ───────────────────────────────────────────
class HaniiPipeline:
    def __init__(self, user_id: str = USER_ID):
        self.user_id = user_id
        self.history = []
        self.relationship = RelationshipManager(user_id)

    def chat(self, user_input: str) -> str:
        rel = self.relationship.get()
        print(f"\n{'─'*40}")

        # ── 1. STATE + PLANNER 통합 호출 ──
        combined_prompt = build_combined_prompt(user_input, rel, self.history)
        combined_raw = call_api(
            STATE_MODEL,
            "너는 Hanii의 STATE_ENGINE이자 PLANNER다. JSON만 반환한다.",
            combined_prompt,
            temperature=0.2,
            max_tokens=300
        )
        combined = parse_json(combined_raw)
        if not combined:
            combined = {
                "emotion_state": {"primary": "neutral", "intensity": 0.5, "suppressed": False},
                "personality_mode": "playful",
                "intent": "random",
                "silence_flag": "none",
                "action_type": "tease",
                "tone": "sharp",
                "followup": False
            }

        state = {
            "emotion_state": combined.get("emotion_state", {"primary": "neutral", "intensity": 0.5, "suppressed": False}),
            "personality_mode": combined.get("personality_mode", "playful"),
            "intent": combined.get("intent", "random"),
            "attention_focus": "viewer",
            "silence_flag": combined.get("silence_flag", "none"),
            "relationship_stage": rel["stage"],
            "trust_score": rel["trust_score"]
        }
        plan = {
            "action_type": combined.get("action_type", "tease"),
            "strategy": combined.get("action_type", "tease"),
            "tone": combined.get("tone", "sharp"),
            "followup": combined.get("followup", False)
        }

        print(f"  STATE  → {state['emotion_state']['primary']} / {state['personality_mode']} / {state['intent']}")
        print(f"  PLAN   → {plan['action_type']} / {plan['tone']}")

        # ── 침묵 처리 ──
        if state.get("silence_flag") == "overwhelmed":
            self._update_and_save(user_input, "....", state, plan, rel)
            return "...."
        if state.get("silence_flag") == "rejection":
            self._update_and_save(user_input, "", state, plan, rel)
            return ""

        # ── 3. SPEAKER ──
        speaker_system = build_speaker_system(state, plan)

        messages = [{"role": "system", "content": speaker_system}]
        messages.extend(self.history[-10:])
        messages.append({"role": "user", "content": user_input})

        response_obj = client.chat.completions.create(
            model=FINETUNED_MODEL,
            messages=messages,
            temperature=0.85,
            max_tokens=300
        )
        response = response_obj.choices[0].message.content.strip()
        print(f"  SPEAKER → 완료")

        # ── 4. 저장 ──
        self._update_and_save(user_input, response, state, plan, rel)

        return response

    def _update_and_save(self, user_input, response, state, plan, rel):
        # 관계 업데이트
        self.relationship.update(
            state.get("intent", "random"),
            state.get("emotion_state", {}).get("primary", "neutral")
        )

        # 히스토리
        self.history.append({"role": "user", "content": user_input})
        if response:
            self.history.append({"role": "assistant", "content": response})

        # 에피소드 저장
        ep_id = save_episode(self.user_id, user_input, response, state, plan)
        print(f"  EPISODE → {ep_id}")


# ── 실행 ──────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  HANII PIPELINE v2")
    print(f"  모델: {FINETUNED_MODEL}")
    print(f"  유저: {USER_ID}")
    print("  종료: quit / exit")
    print("=" * 50)

    pipeline = HaniiPipeline(user_id=USER_ID)

    while True:
        try:
            user_input = input("\n에릭: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "종료"]:
                print("종료합니다.")
                break

            response = pipeline.chat(user_input)
            print(f"\nHanii: {response}")

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
