"""
Smart Librarian - Streamlit Web UI Entrypoint

This script launches the Smart Librarian application with a Streamlit interface.
Features included in the UI:
- Chat interface backed by Retrieval-Augmented Generation (RAG).
- Book recommendation with automatic title extraction.
- Automatic tool call to fetch full book summaries.
- Optional DALL·E 3 illustration generation for the recommended title.
- Text-to-Speech (TTS) playback of assistant responses (via pyttsx3).
- Profanity filter on user input.
- Chat history persistence within the session.
- Reset button to clear the conversation state.

Run this file with:
    streamlit run app.py
"""

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
    st.subheader("Settings")

    model = st.selectbox("Models", ALLOWED_MODELS, index=0)

    # Reset chat button: clears all chat-related session state, then reruns the app
    if st.button("♻️ Reset chat", help="Clean up conversation history and artifacts"):
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
    st.markdown("**Examples:**")
    st.markdown("- „Vreau o carte despre prietenie și magie”")
    st.markdown("- „Vreau o carte despre spiritualitate”")
    st.markdown("- „Ce recomanzi pentru povesti de razboi?”")
    st.markdown("- „Ce este 1984?”")
    st.markdown("- „Ai o carte despre animale?”")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def render_history() -> None:
    """
    Render chat history as user/assistant bubbles.
    """
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
prompt = st.chat_input("Write your query here…")

if prompt:
    # 1) Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Profanity/safety gate BEFORE calling the LLM
    if not is_clean(prompt):
        reply = "Te rog pastreaza un limbaj respectuos. Iti pot recomanda carti pe orice tema."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        # 3) Call the agent (RAG → recommend title → tool: summary)
        with st.chat_message("assistant"):
            with st.spinner("Gandesc…"):
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
    "🖼️ Generate illustration",
    disabled=not bool(title),
    key="btn_image",
    help="Generate an illustration inspired by the recommended title"
):
    if not title:
        st.info("Nu am putut detecta titlul din raspuns.")
    else:
        try:
            # BUGFIX: use correct keyword 'language' (not 'lang')
            path = generate_book_image(title, themes=None, size="1024x1024", lang="ro")
            st.session_state.last_image_path = path
            st.success(f"Generated image: {path}")
        except Exception as e:
            st.error(f"Image generation error: {e}")

# Show last generated image, if any
if st.session_state.last_image_path:
    st.image(st.session_state.last_image_path, caption=f"„{title or ''}” – ilustration DALL·E 3")

# B) Text-to-Speech for last assistant reply
if col2.button(
    "🔊 Read the answer",
    disabled=not bool(reply),
    key="btn_tts",
    help="Play audio of the assistant's last response"
):
    try:
        wav_path = tts_with_pyttsx3_to_wav(reply)
        st.session_state.last_tts_path = wav_path
    except Exception as e:
        st.error(f"TTS error: {e}")

# Audio player (if we have a generated WAV)
if st.session_state.last_tts_path:
    st.audio(st.session_state.last_tts_path)
