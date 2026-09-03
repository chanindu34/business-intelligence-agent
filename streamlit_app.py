import streamlit as st
from app import run_agent

st.set_page_config(page_title="Business Intelligence Agent", page_icon="🤖")

st.title("Business Intelligence Agent")
st.caption(
    "Ask a math question or a question about John Keells Holdings' Annual Report — "
    "the agent decides which tool to use."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("Try an example:")
    examples = [
        "What is 128 divided by 4?",
        "What are the biggest risks facing the company?",
        "How many risks does the report mention?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, use_container_width=True, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                with st.spinner("Thinking..."):
                    answer = run_agent(ex)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_agent(user_question)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})