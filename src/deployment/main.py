import streamlit as st
import torch
import time

from src.model.roberta import CustomRobertaForSequenceClassification
from src.data.DataPreprocessor import DataPreprocessor
from src.model import generate_response


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomRobertaForSequenceClassification(num_labels=20000).to(device)
model.load_model()

def response_gen(_prompt):
    # Trim input text and tokenize
    _input = DataPreprocessor(_prompt)
    _input.trim_patient_description()
    tokenized_input = _input.tokenize_data()
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    # get model
    predictions = model.predict(tokenized_input)

    # TODO: get papers according to predictions
    for papers in predictions:
        pass
    papers = ""

    # summarize papers
    _prompt = DataPreprocessor().summarize_text_with_format(papers)
    _response = generate_response(_prompt)

    # type out response word by word
    for word in _response.split():
        yield word + " "
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("insert patient data"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_gen(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
