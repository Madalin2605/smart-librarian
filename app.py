import os, sys, re
import streamlit as st
import pyttsx3

# ---- Make project modules importable (chatbot/, tools/, etc.) ----
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- Reuse your existing modules ----
from chatbot.retriever import populate_chroma              # Chroma init (idempotent)  [RAG]
from chatbot.agent import run_agent                        # Your agent (RAG + summary tool)  :contentReference[oaicite:3]{index=3}
from tools.language_filter import is_clean                 # Profanity gate (better_profanity)  :contentReference[oaicite:4]{index=4}
from tools.image_generator import generate_book_image            # DALL·E 3 helper you added

ALLOWED_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano")

# --------------------- Streamlit page setup ----------------------
st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="centered")
st.title("📚 Smart Librarian")
st.caption("RAG + recomandare + rezumat + imagine + TTS (pyttsx3) • modele: 4o-mini / 4.1-mini / 4.1-nano")

# --------------------- One-time init ----------------------------
@st.cache_resource
def _init_once():
    # Seed Chroma once (safe if already populated)  :contentReference[oaicite:5]{index=5}
    try:
        populate_chroma()
    except Exception as e:
        st.toast(f"Chroma init: {e}", icon="⚠️")

    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/audio", exist_ok=True)
    return True

_init_once()

# --------------------- Session state ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content}]
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""        # last assistant text
if "last_title" not in st.session_state:
    st.session_state.last_title = None      # parsed title
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None
if "last_tts_path" not in st.session_state:
    st.session_state.last_tts_path = None

# --------------------- Sidebar ----------------------------------
with st.sidebar:
    st.subheader("Setări")
    model = st.selectbox("Model", ALLOWED_MODELS, index=0)
    st.markdown("---")
    st.markdown("**Exemple:**")
    st.markdown("- „Vreau o carte despre prietenie și magie”")
    st.markdown("- „Ce recomanzi pentru povești de război?”")

# --------------------- Helpers ----------------------------------
def render_history():
    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

def extract_title(reply: str) -> str | None:
    m = re.search(r"Recomandare:\s*(.+)", reply, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: first non-empty line (your agent can return only title sometimes)  :contentReference[oaicite:6]{index=6}
    for line in reply.splitlines():
        s = line.strip()
        if s:
            return s
    return None

def tts_with_pyttsx3_to_wav(text: str) -> str:
    """Generate WAV with pyttsx3 and return path; played with st.audio()."""
    path = os.path.join("outputs", "audio", "tts_output.wav")
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    # Optionally choose a Romanian voice if installed:
    # for v in engine.getProperty("voices"):
    #     if "Romanian" in v.name or "Irina" in v.id:
    #         engine.setProperty("voice", v.id); break
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path

# --------------------- UI: history + input ----------------------
render_history()
prompt = st.chat_input("Scrie mesajul tău…")

if prompt:
    # show user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # profanity gate BEFORE LLM  :contentReference[oaicite:7]{index=7}
    if not is_clean(prompt):
        reply = "🙏 Te rog păstrează un limbaj respectuos. Îți pot recomanda cărți pe orice temă."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        # call agent: RAG → title → (tool) summary  :contentReference[oaicite:8]{index=8}
        with st.chat_message("assistant"):
            with st.spinner("Gândesc…"):
                try:
                    reply = run_agent(prompt, model=model)
                except Exception as e:
                    reply = f"❌ Eroare: {e}"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # store last reply + parsed title for buttons
        st.session_state.last_reply = reply
        st.session_state.last_title = extract_title(reply)
        st.session_state.last_image_path = None
        st.session_state.last_tts_path = None

# --------------------- Actions (always rendered) ----------------
st.markdown("")  # spacing

col1, col2 = st.columns(2)
title = st.session_state.last_title
reply = st.session_state.last_reply

# Generate image (DALL·E 3)
if col1.button(
    "🖼️ Generează ilustrație",
    disabled=not bool(title),
    key="btn_image",
    help="DALL·E 3 pentru titlul recomandat"
):
    if not title:
        st.info("Nu am putut detecta titlul din răspuns.")
    else:
        try:
            path = generate_book_image(title, themes=None, size="1024x1024", lang="ro")
            st.session_state.last_image_path = path
            st.success(f"Imagine generată: {path}")
        except Exception as e:
            st.error(f"Eroare generare imagine: {e}")

if st.session_state.last_image_path:
    st.image(st.session_state.last_image_path, caption=f"„{title or ''}” – ilustrație DALL·E 3")

# TTS (pyttsx3 → WAV → audio player)
if col2.button(
    "🔊 Citește răspunsul",
    disabled=not bool(reply),
    key="btn_tts",
    help="Redă audio ultimul răspuns"
):
    try:
        wav = tts_with_pyttsx3_to_wav(reply)
        st.session_state.last_tts_path = wav
    except Exception as e:
        st.error(f"Eroare TTS: {e}")

if st.session_state.last_tts_path:
    st.audio(st.session_state.last_tts_path)
