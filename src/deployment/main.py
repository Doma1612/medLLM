import streamlit as st
import time
from src.models import mistral

st.write("""
# Hello World
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

tokenizer, model = mistral.load_model()

# currently mirrors promt
def response_gen():
    response = mistral.generate_response(model, tokenizer, prompt)

    # type out response word by word
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# React to user input
if prompt := st.chat_input("insert patient data"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_gen())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})