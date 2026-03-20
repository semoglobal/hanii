"""
HANII API SERVER
================
FastAPI 기반 서버. Railway 배포용.
pipeline_v2.py 로직을 API 엔드포인트로 래핑.

엔드포인트:
  POST /chat        → 대화
  GET  /health      → 서버 상태 확인
  GET  /history     → 대화 히스토리
  POST /reset       → 대화 초기화
"""

import os
import json
import re
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Hanii API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 설정 ──────────────────────────────────────────────────────
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "ft:gpt-4o-mini-2024-07-18:hdd:hanii:DLZyblME")
STATE_MODEL     = "gpt-4o-mini"
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

# ── 구조 파일 로드 ─────────────────────────────────────────────
print("[로딩] 구조 파일 로드 중...")
SYSTEM_PROMPT      = load_txt("13_RUNTIME_PROMPTS/system_prompt.txt")
EMOTION_RULES      = load_json("11_STATE_ENGINE/emotion_state_rules.json")
INTENT_MAP         = load_json("11_STATE_ENGINE/intent_state_map.json")
SILENCE_RULES      = load_json("11_STATE_ENGINE/silence_rules.json")
PLANNER_RULES      = load_json("12_PLANNER/planner_rules.json")
FOLLOWUP_RULES     = load_json("12_PLANNER/followup_rules.json")
REPAIR_SEQUENCES   = load_json("12_PLANNER/repair_sequences.json")
PERSONALITY_MODES  = load_json("17_PERSONALITY/personality_modes.json")
TONE_STYLE         = load_json("17_PERSONALITY/tone_style.json")
HUMOR_RULES        = load_json("17_PERSONALITY/humor_rules.json")
REACTION_TYPES     = load_json("18_REACTION_ENGINE/reaction_types.json")
REACTION_RULES     = load_json("18_REACTION_ENGINE/reaction_rules.json")
CURIOSITY_RULES    = load_json("19_CURIOSITY_ENGINE/curiosity_rules.json")
QUESTION_TEMPLATES = load_json("19_CURIOSITY_ENGINE/question_templates.json")
KNOWLEDGE_BASE     = load_json("09_KNOWLEDGE/KNOWLEDGE_BASE.json")
print("[로딩] 완료")

# ── 세션 관리 (메모리) ─────────────────────────────────────────
sessions = {}

def get_session(user_id: str) -> dict:
    if user_id not in sessions:
        sessions[user_id] = {
            "history": [],
            "relationship": {
                "user_id": user_id,
                "stage": 1,
                "trust_score": 0.0,
                "interaction_count": 0,
                "recent_interaction": "none",
                "emotional_depth_score": 0.0
            }
        }
    return sessions[user_id]

def update_relationship(rel: dict, intent: str, emotion: str):
    rel["interaction_count"] += 1
    rel["recent_interaction"] = intent
    delta = {
        "genuine": 0.05, "emotional": 0.10, "flatter": 0.02,
        "provoke": -0.01, "random": 0.01, "romantic": 0.03, "test": 0.01
    }.get(intent, 0)
    if emotion in ["touched", "overwhelmed"]:
        delta += 0.08
        rel["emotional_depth_score"] = min(1.0, rel["emotional_depth_score"] + 0.1)
    rel["trust_score"] = round(max(0.0, min(1.0, rel["trust_score"] + delta)), 3)
    s, c, e = rel["trust_score"], rel["interaction_count"], rel["emotional_depth_score"]
    if s >= 0.8 and c >= 20 and e >= 0.3:
        rel["stage"] = 4
    elif s >= 0.5 and c >= 10:
        rel["stage"] = 3
    elif s >= 0.2 and c >= 3:
        rel["stage"] = 2

# ── 에피소드 저장 ─────────────────────────────────────────────
def save_episode(user_id, user_input, response, state, plan):
    ep_dir = os.path.join(BASE_DIR, "04_EPISODIC_MEMORY")
    os.makedirs(ep_dir, exist_ok=True)
    ep_id = f"EP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    episode = {
        "episode_id": ep_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "importance": round(min(0.99, state.get("trust_score", 0.1) + 0.3), 2),
        "tags": [state.get("intent", "none"), state.get("personality_mode", "playful")],
        "state_snapshot": state,
        "plan_snapshot": plan,
        "Dialogue": [
            {"speaker": user_id, "text": user_input},
            {"speaker": "Hanii", "text": response}
        ]
    }
    with open(os.path.join(ep_dir, f"{ep_id}.json"), "w", encoding="utf-8") as f:
        json.dump(episode, f, ensure_ascii=False, indent=2)
    return ep_id

# ── 프롬프트 빌더 ─────────────────────────────────────────────
def build_state_prompt(user_input, rel, history):
    emotion_samples = {k: v.get("sample", "") for k, v in EMOTION_RULES.get("states", {}).items() if v.get("sample")}
    recent = "\n".join([f"{'유저' if m['role']=='user' else 'Hanii'}: {m['content']}" for m in history[-6:]]) or "없음"
    return f"""유저: {rel['user_id']} / 단계: {rel['stage']} / 신뢰: {rel['trust_score']:.2f} / 횟수: {rel['interaction_count']}

최근 대화:
{recent}

현재 입력: {user_input}

의도 판단 예시:
- "나 혼자인데 왜 너희들이야" → intent: test
- "AI가 뭘 알아" → intent: test
- "힘들어", "외로워" → intent: emotional
- "좋아하는 사람 있어?" → intent: romantic
- "안녕", "뭐해" → intent: genuine
- "치킨이랑 피자" → intent: random
- 도발, 장난 → intent: provoke
- 칭찬, 어리게 봐주기 → intent: flatter

반드시 JSON만:
{{"emotion_state":{{"primary":"...","intensity":0.5,"suppressed":false}},"personality_mode":"playful","intent":"genuine","attention_focus":"viewer","silence_flag":"none"}}"""

def build_planner_prompt(user_input, state, rel):
    rules = PLANNER_RULES.get("decision_matrix", {}).get("rules", [])
    return f"""STATE: {json.dumps(state, ensure_ascii=False)}
관계: stage={rel['stage']}, trust={rel['trust_score']:.2f}
입력: {user_input}
규칙: {json.dumps(rules[:8], ensure_ascii=False)}

반드시 JSON만:
{{"action_type":"tease","strategy":"...","tone":"sharp","followup":false}}"""

def build_speaker_system(state, plan):
    mode = state.get("personality_mode", "playful")
    action = plan.get("action_type", "tease")
    emotion = state.get("emotion_state", {}).get("primary", "neutral")
    mode_rules = PERSONALITY_MODES.get("modes", {}).get(mode, {}).get("behavior", [])
    reaction = REACTION_TYPES.get("types", {}).get(action, {})
    tone_forbidden = TONE_STYLE.get("forbidden_expressions", {}).get("list", [])
    collapse = REACTION_RULES.get("character_stability_rules", {}).get("collapse_examples", [])

    extra = ""
    if mode == "playful":
        extra += f"\n유머패턴: {list(HUMOR_RULES.get('humor_patterns', {}).keys())}"
    if mode == "curious" or action == "ask":
        extra += f"\n질문체이닝: {json.dumps(QUESTION_TEMPLATES.get('chaining_rule', {}), ensure_ascii=False)}"
    if mode == "sincere" or emotion in ["touched", "overwhelmed", "anxious"]:
        extra += f"\nsincere말투: {json.dumps(TONE_STYLE.get('sincere_mode_tone', {}).get('characteristics', []), ensure_ascii=False)}"

    return f"""{SYSTEM_PROMPT}

## 현재 상태
감정: {emotion} / 모드: {mode} / 단계: {state.get('relationship_stage',1)} / 신뢰: {state.get('trust_score',0):.2f}

## 행동계획
action: {action} / strategy: {plan.get('strategy','')} / tone: {plan.get('tone','sharp')}

## 모드규칙
{mode_rules}{extra}

## 반응타입
{json.dumps(reaction, ensure_ascii=False)}

## 금지표현
{json.dumps(tone_forbidden, ensure_ascii=False)}

## 붕괴방지
{json.dumps(collapse, ensure_ascii=False)}

## 지식베이스
{json.dumps(KNOWLEDGE_BASE.get('hanii_self_knowledge', {}), ensure_ascii=False)}
{json.dumps(KNOWLEDGE_BASE.get('streaming_platforms', {}), ensure_ascii=False)}"""

def parse_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return {}

# ── 핵심 파이프라인 (병렬 처리) ───────────────────────────────
async def run_pipeline(user_input: str, user_id: str) -> dict:
    session = get_session(user_id)
    rel = session["relationship"]
    history = session["history"]

    # STATE + PLANNER 병렬 실행
    state_prompt = build_state_prompt(user_input, rel, history)

    async def call_state():
        r = await client.chat.completions.create(
            model=STATE_MODEL,
            messages=[
                {"role": "system", "content": "너는 Hanii의 STATE_ENGINE. JSON만 반환."},
                {"role": "user", "content": state_prompt}
            ],
            temperature=0.2, max_tokens=250
        )
        return parse_json(r.choices[0].message.content)

    # STATE 먼저 실행 (PLANNER는 STATE 결과 필요)
    state = await call_state()
    if not state:
        state = {"emotion_state": {"primary": "neutral", "intensity": 0.5, "suppressed": False},
                 "personality_mode": "playful", "intent": "random",
                 "attention_focus": "viewer", "silence_flag": "none"}

    state["relationship_stage"] = rel["stage"]
    state["trust_score"] = rel["trust_score"]

    # 침묵 처리
    if state.get("silence_flag") == "overwhelmed":
        update_relationship(rel, state["intent"], state["emotion_state"]["primary"])
        session["history"].extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": "...."}])
        return {"response": "....", "state": state, "plan": {}, "episode_id": save_episode(user_id, user_input, "....", state, {})}

    if state.get("silence_flag") == "rejection":
        return {"response": "", "state": state, "plan": {}, "episode_id": ""}

    # PLANNER 실행
    planner_prompt = build_planner_prompt(user_input, state, rel)
    plan_r = await client.chat.completions.create(
        model=STATE_MODEL,
        messages=[
            {"role": "system", "content": "너는 Hanii의 PLANNER. JSON만 반환."},
            {"role": "user", "content": planner_prompt}
        ],
        temperature=0.2, max_tokens=150
    )
    plan = parse_json(plan_r.choices[0].message.content) or {"action_type": "tease", "strategy": "현실 던지기", "tone": "sharp", "followup": False}

    # SPEAKER 실행
    speaker_system = build_speaker_system(state, plan)
    messages = [{"role": "system", "content": speaker_system}]
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": user_input})

    speaker_r = await client.chat.completions.create(
        model=FINETUNED_MODEL,
        messages=messages,
        temperature=0.85,
        max_tokens=300
    )
    response = speaker_r.choices[0].message.content.strip()

    # 저장
    update_relationship(rel, state["intent"], state["emotion_state"]["primary"])
    session["history"].extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": response}])
    ep_id = save_episode(user_id, user_input, response, state, plan)

    return {
        "response": response,
        "state": {
            "emotion": state["emotion_state"]["primary"],
            "mode": state["personality_mode"],
            "intent": state["intent"],
            "stage": rel["stage"],
            "trust": rel["trust_score"]
        },
        "plan": {"action": plan["action_type"], "tone": plan["tone"]},
        "episode_id": ep_id
    }

# ── API 엔드포인트 ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "eric_ha"

class ResetRequest(BaseModel):
    user_id: Optional[str] = "eric_ha"

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있어.")
    try:
        result = await run_pipeline(req.message, req.user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": FINETUNED_MODEL}

@app.get("/history/{user_id}")
async def history(user_id: str):
    session = get_session(user_id)
    return {
        "history": session["history"][-20:],
        "relationship": session["relationship"]
    }

@app.post("/reset")
async def reset(req: ResetRequest):
    if req.user_id in sessions:
        sessions[req.user_id]["history"] = []
    return {"status": "reset", "user_id": req.user_id}

@app.get("/")
async def root():
    return {"message": "Hanii API Server", "version": "1.0"}
