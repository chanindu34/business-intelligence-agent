"""Streamlit chat UI: shows each tool the agent decides to call, live."""

import json
import os
import re
import time

import streamlit as st

st.set_page_config(page_title="BI Agent", layout="centered", initial_sidebar_state="collapsed")

# Streamlit Cloud secrets -> environment, before anything reads the key.
for name in ("GEMINI_API_KEY", "BI_RERANKER", "BI_MODELS"):
    try:
        if not os.environ.get(name) and name in st.secrets:
            os.environ[name] = str(st.secrets[name])
    except Exception:
        pass

from agent import run_agent  # noqa: E402
from service import get_client, get_embedder, get_index  # noqa: E402

SAMPLES = [
    "What share of the Group's recurring EBITDA came from the Retail industry group?",
    "How much did Group EBITDA grow in 2025/26?",
    "By how much in rupees did Group EBITDA increase from 2024/25 to 2025/26?",
    "What is 15% of 2,400?",
]


def escape_md(text: str) -> str:
    """Show user text literally: '9**9**9' must not render as bold '999'."""
    return re.sub(r"([\\`*_{}\[\]<>()#+\-.!|~$])", r"\\\1", text)


def plain_answer(text: str) -> str:
    """Stop '$...$' in model output rendering as LaTeX maths."""
    return text.replace("$", r"\$")


@st.cache_resource
def load():
    return get_client(), get_index(), get_embedder()


try:
    client, index, embed = load()
except Exception as e:
    st.error(f"Could not start: {e}")
    st.stop()


def describe(step) -> str:
    r, a = step["result"], step["args"]
    if step["tool"] == "calculate":
        return f"Calculated `{a.get('expression')}` = **{r['result']}**" if r.get("success") else f"Calculation failed: {r.get('error')}"
    if step["tool"] == "search_knowledge_base":
        pages = sorted({p["page"] for p in r.get("passages", []) if p.get("page")})
        return f"Searched the report for *{a.get('query')}*: {len(r.get('passages', []))} passages, pages {pages}"
    return f"{step['tool']}: {r.get('error', 'done')}"


def render_meta(meta) -> None:
    badges = []
    tools = meta.get("tools_used", [])
    if not tools:
        badges.append(":gray-badge[No tools needed]")
    if "search_knowledge_base" in tools:
        badges.append(":blue-badge[Report search]")
    if "calculate" in tools:
        badges.append(":violet-badge[Calculator]")
    if len(tools) > 1:
        badges.append(f":green-badge[{len(tools)} tool calls]")
    if meta.get("stopped") not in (None, "answered"):
        badges.append(f":orange-badge[Stopped: {meta['stopped']}]")
    if meta.get("seconds"):
        badges.append(f":gray-badge[{meta['seconds']:.1f}s]")
    st.markdown(" ".join(badges))
    if meta.get("steps"):
        with st.expander(f"Agent steps ({len(meta['steps'])})"):
            for i, s in enumerate(meta["steps"], 1):
                st.markdown(f"**{i}.** {s}")


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(escape_md(question))
    with st.chat_message("assistant"):
        started = time.time()
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.status("Thinking...", expanded=True) as status:
            def on_step(ev):
                if ev["event"] == "tool_start":
                    status.write(f"Calling `{ev['tool']}` with `{json.dumps(ev['args'])}`")
                elif ev["event"] == "tool_end":
                    status.write(describe(ev))
                elif ev["event"] == "model_failover":
                    status.write(f"Model unavailable, switching: {ev['detail']}")
                elif ev["event"] == "verifier":
                    status.write("Check: the answer had a sum the calculator did not run, sending it back")

            result = run_agent(question, client, index=index, history=history, embed=embed, on_step=on_step)
            n = len(result.steps)
            status.update(label=f"{n} tool call{'s' if n != 1 else ''} in {time.time() - started:.1f}s" if n
                          else "Answered without tools", state="complete", expanded=False)
        st.markdown(plain_answer(result.answer))
        meta = {"tools_used": result.tools_used, "stopped": result.stopped, "seconds": result.seconds,
                "steps": [describe(s) for s in result.steps]}
        render_meta(meta)
    st.session_state.messages.append({"role": "assistant", "content": result.answer, "meta": meta})


st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'>Business Intelligence Agent</h1>"
    "<p style='text-align: center; color: gray; font-size: 1.1rem; margin-top: 0;'>"
    "for John Keells Holdings · searches the report, calculates, or both</p>",
    unsafe_allow_html=True,  # static text only
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.messages:
    _, right = st.columns([5, 1])
    right.button("New chat", on_click=lambda: st.session_state.update(messages=[]), use_container_width=True)
elif "pending" not in st.session_state:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLES):
        cols[i % 2].button(q, key=f"s{i}", use_container_width=True,
                           on_click=lambda q=q: st.session_state.update(pending=q))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(escape_md(m["content"]) if m["role"] == "user" else plain_answer(m["content"]))
        if m.get("meta"):
            render_meta(m["meta"])

typed = st.chat_input("Ask about JKH's 2025/26 results, or give me a calculation...", max_chars=500)
question = (typed or st.session_state.pop("pending", None) or "").strip()
if question:
    ask(question)
