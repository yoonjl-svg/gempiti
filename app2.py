import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from openai import OpenAI
import google.generativeai as genai


# ---------------------------
# Config
# ---------------------------
st.set_page_config(layout="wide", page_title="GemPiTi v1")
st.title("GemPiTi v1 (Shared)")

MAX_INPUT_CHARS = 2000
COOLDOWN_SECONDS = 4          # 세션 내 연속 호출 최소 간격
MAX_CALLS_PER_SESSION = 50    # 세션 내 최대 호출 수
OVERALL_TIMEOUT_SECONDS = 90  # 한 번 질문에서 전체 대기 최대 시간(멈춘 것처럼 보이는 상황 방지)

DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"


# ---------------------------
# Secrets (server-side only)
# ---------------------------
def load_secrets():
    # st.secrets가 없는 로컬 환경도 대비
    try:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        app_password = st.secrets.get("APP_PASSWORD", "")
        openai_model = st.secrets.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        gemini_model = st.secrets.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    except Exception:
        openai_key = ""
        gemini_key = ""
        app_password = ""
        openai_model = DEFAULT_OPENAI_MODEL
        gemini_model = DEFAULT_GEMINI_MODEL

    return openai_key, gemini_key, app_password, openai_model, gemini_model


OPENAI_API_KEY, GEMINI_API_KEY, APP_PASSWORD, OPENAI_MODEL, GEMINI_MODEL = load_secrets()

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    st.error("서버 시크릿이 설정되지 않았습니다. OPENAI_API_KEY, GEMINI_API_KEY를 Secrets에 추가하세요.")
    st.stop()


# ---------------------------
# Auth gate (simple password)
# ---------------------------
def is_authed() -> bool:
    return st.session_state.get("authed", False)

with st.sidebar:
    st.header("접속")

    if APP_PASSWORD:
        pw = st.text_input("Password", type="password", key="pw_input")
        if st.button("Unlock"):
            if pw == APP_PASSWORD:
                st.session_state["authed"] = True
                st.success("접속 허용")
            else:
                st.session_state["authed"] = False
                st.error("비밀번호가 틀렸습니다.")
    else:
        st.session_state["authed"] = True
        st.info("APP_PASSWORD가 설정되지 않아 공개 모드입니다.")

    st.divider()
    st.header("상태")
    st.write(f"OpenAI model: {OPENAI_MODEL}")
    st.write(f"Gemini model: {GEMINI_MODEL}")

if not is_authed():
    st.stop()


# ---------------------------
# Session state for guardrails
# ---------------------------
if "last_call_ts" not in st.session_state:
    st.session_state["last_call_ts"] = 0.0
if "call_count" not in st.session_state:
    st.session_state["call_count"] = 0


def enforce_guardrails(user_text: str):
    if len(user_text) > MAX_INPUT_CHARS:
        raise ValueError(f"입력이 너무 깁니다. {MAX_INPUT_CHARS}자 이하로 줄여주세요.")

    now = time.time()
    if now - st.session_state["last_call_ts"] < COOLDOWN_SECONDS:
        remain = COOLDOWN_SECONDS - (now - st.session_state["last_call_ts"])
        raise ValueError(f"호출 간격이 너무 짧습니다. {remain:.1f}초 후 다시 시도하세요.")

    if st.session_state["call_count"] >= MAX_CALLS_PER_SESSION:
        raise ValueError("세션 호출 한도를 초과했습니다. 새로고침 후 다시 시도하거나, 한도를 조정하세요.")


# ---------------------------
# Client/model caching (편의 우선: 키를 인자로 유지)
# ---------------------------
@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

@st.cache_resource
def configure_gemini_once(api_key: str) -> None:
    # configure는 전역 성격이므로, 동일 키에 대해 1회만 실행되도록 캐싱
    genai.configure(api_key=api_key)

@st.cache_resource
def get_gemini_model(model_name: str):
    return genai.GenerativeModel(model_name)


openai_client = get_openai_client(OPENAI_API_KEY)
configure_gemini_once(GEMINI_API_KEY)
gemini_model = get_gemini_model(GEMINI_MODEL)


# ---------------------------
# AI call functions
# ---------------------------
def ask_openai_responses(prompt: str) -> str:
    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    return getattr(resp, "output_text", "") or ""

def ask_gemini(prompt: str) -> str:
    r = gemini_model.generate_content(prompt)
    return getattr(r, "text", "") or ""


# ---------------------------
# UI
# ---------------------------
prompt = st.chat_input("질문을 입력하세요.")

if prompt:
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    try:
        enforce_guardrails(prompt)
    except Exception as e:
        st.warning(str(e))
        st.stop()

    # guardrail bookkeeping
    st.session_state["last_call_ts"] = time.time()
    st.session_state["call_count"] += 1

    st.write(f"**질문:** {prompt}")
    st.caption(f"세션 호출: {st.session_state['call_count']} / {MAX_CALLS_PER_SESSION}")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("GPT 5.2")
        gpt_box = st.empty()
        gpt_box.info("으음...")

    with col2:
        st.subheader("Gemini 3.0")
        gem_box = st.empty()
        gem_box.info("으음...")

    start_ts = time.time()

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(ask_openai_responses, prompt): ("GPT 5.2", gpt_box),
            ex.submit(ask_gemini, prompt): ("Gemini 3.0", gem_box),
        }

        done = set()

        try:
            for f in as_completed(futures, timeout=OVERALL_TIMEOUT_SECONDS):
                name, box = futures[f]
                elapsed = time.time() - start_ts

                try:
                    text = f.result()
                    box.success(f"완료 ({elapsed:.1f}s)")
                    box.write(text if text else "(빈 응답)")
                except Exception as e:
                    box.error(f"{name} 오류 ({elapsed:.1f}s)")
                    box.write(f"[{name} ERROR] {type(e).__name__}: {e}")

                done.add(f)

        except TimeoutError:
            # 전체 타임아웃: 아직 안 끝난 쪽은 타임아웃 처리
            for f, (name, box) in futures.items():
                if f not in done:
                    box.error(f"{name} 시간 초과 ({OVERALL_TIMEOUT_SECONDS}s)")

                    box.write(f"[{name} ERROR] Timeout")

