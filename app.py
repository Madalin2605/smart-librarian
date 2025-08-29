# Streamlit UI for Smart Librarian: chat (RAG), summary tool calling, DALL·E image gen, and TTS (pyttsx3)

import os
import re
import sys
import streamlit as st
import pyttsx3

# ──────────────────────────────────────────────────────────────────────────────
# Import project modules (ensure project root is on sys.path)
# ──────────────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chatbot.retriever import populate_chroma            
from chatbot.agent import run_agent                      
from tools.language_filter import is_clean               
from tools.image_generator import generate_book_image   


# ──────────────────────────────────────────────────────────────────────────────
# Constants / Settings
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano")  
IMAGES_DIR = "outputs/images"
AUDIO_DIR = "outputs/audio"


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="centered")
st.title("📚 Smart Librarian")
st.caption("RAG + recomandare + rezumat + ilustrație (DALL·E 3) + TTS (pyttsx3) • modele: 4o-mini / 4.1-mini / 4.1-nano")


# ──────────────────────────────────────────────────────────────────────────────
# One-time initialization (cached): seed Chroma, ensure output folders exist
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _init_once() -> bool:
    try:
        populate_chroma() 
    except Exception as e:
        st.toast(f"Chroma init: {e}", icon="⚠️")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    return True

_init_once()


# ──────────────────────────────────────────────────────────────────────────────
# Session state (persists across reruns)
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history: [{role: 'user'|'assistant', content: str}]
if "last_reply" not in st.session_state:
    st.session_state.last_reply = ""       
if "last_title" not in st.session_state:
    st.session_state.last_title = None  
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None
if "last_tts_path" not in st.session_state:
    st.session_state.last_tts_path = None


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: model picker, examples, and RESET CHAT
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Setări")

    model = st.selectbox("Model", ALLOWED_MODELS, index=0)

    # Reset chat button: clears all chat-related session state, then reruns the app
    if st.button("♻️ Reset chat", help="Curăță istoricul conversației și artefactele"):
        st.session_state.messages = []
        st.session_state.last_reply = ""
        st.session_state.last_title = None
        st.session_state.last_image_path = None
        st.session_state.last_tts_path = None
        
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("**Exemple:**")
    st.markdown("- „Vreau o carte despre prietenie și magie”")
    st.markdown("- „Ce recomanzi pentru povești de război?”")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def render_history() -> None:
    """Render chat history as user/assistant bubbles."""
    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])


def extract_title(reply: str) -> str | None:
    """
    Extract the recommended title from assistant reply.
    1) Prefer a 'Recomandare: <titlu>' line
    2) Fallback to the first non-empty line
    """
    m = re.search(r"Recomandare:\s*(.+)", reply, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for line in reply.splitlines():
        s = line.strip()
        if s:
            return s
    return None


def tts_with_pyttsx3_to_wav(text: str) -> str:
    """
    Generate a WAV from the given text using pyttsx3 and return the file path.
    Tip: If you have Romanian voices installed on your OS, pick them by name/id.
    """
    path = os.path.join(AUDIO_DIR, "tts_output.wav")
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    engine.save_to_file(text, path)
    engine.runAndWait()
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Main chat UI: show history and input box
# ──────────────────────────────────────────────────────────────────────────────
render_history()
prompt = st.chat_input("Scrie mesajul tău…")

if prompt:
    # 1) Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Profanity/safety gate BEFORE calling the LLM
    if not is_clean(prompt):
        reply = "🙏 Te rog păstrează un limbaj respectuos. Îți pot recomanda cărți pe orice temă."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        # 3) Call the agent (RAG → recommend title → tool: summary)
        with st.chat_message("assistant"):
            with st.spinner("Gândesc…"):
                try:
                    reply = run_agent(prompt, model=model)
                except Exception as e:
                    reply = f"❌ Eroare: {e}"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # 4) Persist parsed artifacts for action buttons
        st.session_state.last_reply = reply
        st.session_state.last_title = extract_title(reply)
        st.session_state.last_image_path = None
        st.session_state.last_tts_path = None


# ──────────────────────────────────────────────────────────────────────────────
# Actions (always visible): Generate image & TTS for last response
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("")  # spacing

col1, col2 = st.columns(2)
title = st.session_state.last_title
reply = st.session_state.last_reply

# A) Generate illustration with DALL·E 3 for the detected title
if col1.button(
    "🖼️ Generează ilustrație",
    disabled=not bool(title),
    key="btn_image",
    help="Generează o ilustrație inspirată de titlul recomandat (DALL·E 3)"
):
    if not title:
        st.info("Nu am putut detecta titlul din răspuns.")
    else:
        try:
            # BUGFIX: use correct keyword 'language' (not 'lang')
            path = generate_book_image(title, themes=None, size="1024x1024", language="ro")
            st.session_state.last_image_path = path
            st.success(f"Imagine generată: {path}")
        except Exception as e:
            st.error(f"Eroare generare imagine: {e}")

# Show last generated image, if any
if st.session_state.last_image_path:
    st.image(st.session_state.last_image_path, caption=f"„{title or ''}” – ilustrație DALL·E 3")

# B) Text-to-Speech for last assistant reply
if col2.button(
    "🔊 Citește răspunsul",
    disabled=not bool(reply),
    key="btn_tts",
    help="Redă audio ultimul răspuns al asistentului"
):
    try:
        wav_path = tts_with_pyttsx3_to_wav(reply)
        st.session_state.last_tts_path = wav_path
    except Exception as e:
        st.error(f"Eroare TTS: {e}")

# Audio player (if we have a generated WAV)
if st.session_state.last_tts_path:
    st.audio(st.session_state.last_tts_path)
