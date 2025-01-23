import streamlit as st
from transformers import pipeline
import pandas as pd
import torch
import time
import os

from src.model.roberta import CustomRobertaForSequenceClassification
from src.data.DataPreprocessor import DataPreprocessor
from src.data.deep_symptom_extraction import SymptomExtractor
from src.model import generate_response
from src.data.deep_symptom_extraction import symptom_list

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Initialize the device
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda")  # NVIDIA GPU with CUDA
    print("Using device: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU (MPS)
    print("Using device: MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using device: CPU")

# initialize roberta model
model = CustomRobertaForSequenceClassification(num_labels=20000).to(device)
model.load_model(ROOT_DIR)

# initialize generator
model_name = "meta-llama/Llama-3.2-1B-Instruct"
generator = pipeline(model=model_name, device=device, torch_dtype=torch.float16, task="summarization")

# get paper data
full_text_papers = pd.read_csv(f"{ROOT_DIR}/data/pmc_patients/processed/full_texts_combined.csv")
full_text_papers_ids = pd.read_csv(f"{ROOT_DIR}/data/pmc_patients/Summary_data/list_of_articles_with_full_text.csv")

def get_list_of_articles(_predictions):
    paper_ids = full_text_papers_ids[full_text_papers_ids["index"].isin(_predictions)]

    _predictions_df = full_text_papers[full_text_papers["PMID"].isin(paper_ids["article"])]
    return _predictions_df["full_text"].tolist()

def response_gen(_prompt):
    # Trim input text and tokenize
    symptom_extractor = SymptomExtractor()
    _prompt = symptom_extractor.extract_symptoms(_prompt)
    _input = DataPreprocessor(_prompt)
    _input.trim_patient_description()
    tokenized_input = _input.tokenize_data()
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    # get model
    # TODO: uncomment next line
    # predictions = model.predict(tokenized_input)
    # TODO: remove test predictions
    predictions = [89]

    # map to paper ids
    list_of_articles = get_list_of_articles(predictions)

    # summarize papers
    _prompt = DataPreprocessor().summarize_text_with_format(list_of_articles)
    _response = generate_response(_prompt, generator, len(list_of_articles))

    # type out response word by word
    for word in _response.split():
        yield word + " "
        time.sleep(0.05)

    torch.cuda.empty_cache()

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
