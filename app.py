"""Streamlit UI for the AI Research Agent."""

import os

# Turn off CrewAI telemetry (must be set before importing crewai).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st

from crew import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 AI Research Agent")
st.caption("CrewAI + Groq (gpt-oss-120b) + DuckDuckGo search")


def get_api_key():
    """Reads the Groq key from Streamlit secrets."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


api_key = get_api_key()
if not api_key:
    st.error(
        "GROQ_API_KEY not found. In Streamlit Cloud open your app → "
        "Settings → Secrets and add:\n\n`GROQ_API_KEY = \"your_key_here\"`"
    )
    st.stop()

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Latest advances in solid-state batteries",
)

if st.button("Generate report", type="primary", disabled=not topic.strip()):
    with st.spinner("Researching... this can take 1-2 minutes."):
        try:
            st.session_state["report"] = run_research(topic.strip(), api_key)
        except Exception as e:
            st.session_state.pop("report", None)
            st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
