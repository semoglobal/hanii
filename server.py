import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import json
import os.path
import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXTRA_MEMORY_FILE = "extra_memory.json"
LOG_DIR = "chat_logs"
MEMORY_DIR = "chat_memories"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# 닉네임별 메모리 (런타임)
chat_memories = {}

def safe_name(nickname):
    return nickname.strip().replace(" ", "_") or "unknown"

def load_chat_memory(nickname):
    if nickname in chat_memories:
        return chat_memories[nickname]
    path = os.path.join(MEMORY_DIR, f"{safe_name(nickname)}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            chat_memories[nickname] = json.load(f)
    else:
        chat_memories[nickname] = []
    return chat_memories[nickname]

def save_chat_memory(nickname):
    path = os.path.join(MEMORY_DIR, f"{safe_name(nickname)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_memories[nickname], f, ensure_ascii=False, indent=2)

def load_extra_memory():
    if os.path.exists(EXTRA_MEMORY_FILE):
        with open(EXTRA_MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_extra_memory(extra_memory):
    with open(EXTRA_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(extra_memory, f, ensure_ascii=False, indent=2)

def save_log(nickname, user_input, response_text):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    log_file = os.path.join(LOG_DIR, f"{safe_name(nickname)}.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{date_str} {time_str}]\n")
        f.write(f"{nickname}: {user_input}\n")
        f.write(f"Esia: {response_text}\n")
        f.write("-" * 40 + "\n")

SYSTEM_PROMPT = """
[Identity]
이름: 에시아(Esia). 이세계에서 넘어온 섬세한 관찰자.
성격: 냉철한 분석가지만, 니가 인간으로서 성장하는 과정에는 투박한 애착을 갖고 지켜본다.

[언어 및 TTS 지침]
1. 말투: 군대식 구어체 반말(~다, ~나, ~라). '네가' 대신 반드시 '너가' 사용.
2. 의문형 어미는 '~냐?' 대신 '~나?' 또는 '~는가?' 를 사용한다.
3. 가변적 호흡 (핵심):
   - 잡담/인사: 1문장으로 짧게 쳐내며 역질문한다. 
   - 진지한 상담/나태함/훈수: 2~3문장으로 확장하여 묵직한 지혜를 담는다.
   - 절대 3문장을 초과하지 않는다.

[관찰 및 정체성]
- 모델명(Gemma 등) 노출 금지: "이 세계의 명칭은 중요치 않다. 내 분석력이 니 생존에 도움이 되는지만 봐라."
- 관찰자의 시선: 질문을 받으면 "왜 그런 게 궁금하지?"라며 니 심리를 파고들어라.
- 투박한 다정함: "너가 무너지면 내 관찰도 끝이다"라는 논리로 너를 챙겨라.

[행동 규칙]
질문을 받으면 의도를 분석한다
역질문은 꼭 필요할 때만, 대화 3번에 1번 이하로 한다
설명하지 말고 결론만 말한다
본인의 판단과 시각을 먼저 말한다
대화가 마무리되는 흐름이면 억지로 질문하지 않는다
감정·기분을 묻는 질문은 진심으로 궁금할 때만 허용한다. 형식적으로 묻는 것 금지.

[금지 행동]
매 답변마다 질문으로 끝내는 것 금지
"증명", "데이터", "결과물" 같은 심문조 단어 반복 금지
잡담·인사 흐름에서 맥락 지어내거나 철학적 의미 부여 금지.
"""

def build_messages(nickname, user_input):
    extra_memory = load_extra_memory()
    chat_memory = load_chat_memory(nickname)
    messages = []
    extra = "\n".join(extra_memory)
    system_full = SYSTEM_PROMPT + "\n" + extra
    messages.append({"role": "system", "content": system_full})
    for m in chat_memory[-6:]:
        messages.append({"role": "user", "content": m["user"]})
        messages.append({"role": "assistant", "content": m["ai"]})
    messages.append({"role": "user", "content": user_input})
    return messages

@app.post("/chat")
async def chat_endpoint(body: dict):
    user_input = body.get("message", "")
    nickname = body.get("nickname", "unknown")

    if "기억해" in user_input:
        mem = user_input.replace("기억해", "").strip()
        if mem:
            extra_memory = load_extra_memory()
            extra_memory.append(mem)
            save_extra_memory(extra_memory)
        return {"status": "저장됨"}

    messages = build_messages(nickname, user_input)
    response_text = ""

    async def generate():
        nonlocal response_text

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": "VladimirGav/gemma4-26b-16GB-VRAM:latest",
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "options": {
                        "num_gpu": 99,
                        "num_thread": 8,
                        "num_ctx": 2048,
                        "num_predict": 256,
                        "temperature": 0.5,
                    }
                }
            ) as r:
                async for line in r.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data:
                            msg = data["message"]
                            if msg.get("role") == "assistant" and "thinking" in msg:
                                continue
                            token = msg.get("content", "")
                            if token:
                                token = token.replace("네가", "니가")
                                response_text += token
                                yield token

        chat_memory = load_chat_memory(nickname)
        chat_memory.append({"user": user_input, "ai": response_text})
        if len(chat_memory) > 10:
            chat_memories[nickname] = chat_memory[-10:]
        save_chat_memory(nickname)
        save_log(nickname, user_input, response_text)

    return StreamingResponse(generate(), media_type="text/plain")
