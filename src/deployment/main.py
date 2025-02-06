import streamlit as st
from transformers import pipeline
import pandas as pd
import torch
import os

from src.model.roberta import CustomRobertaForSequenceClassification
from src.data.DataPreprocessor import DataPreprocessor
from src.model import generate_response

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

# initialize generator
model_name = "meta-llama/Llama-3.2-1B-Instruct"
generator = pipeline(model=model_name, device=device, torch_dtype=torch.float16)

# get paper data
full_text_papers = pd.read_csv(f"{ROOT_DIR}/data/pmc_patients/processed/full_texts_combined.csv")
full_text_papers_ids = pd.read_csv(f"{ROOT_DIR}/notebooks/extended_list_of_articles.csv")

def get_list_of_articles(_predictions):
    paper_ids = full_text_papers_ids[full_text_papers_ids.iloc[:, 0].isin(_predictions)]
    print(paper_ids)
    _predictions_df = full_text_papers[full_text_papers["PMID"].isin(paper_ids["article"])]
    return _predictions_df["full_text"].tolist()

def response_gen(_prompt):
    # Trim input text and tokenize
    data_preprocessor = DataPreprocessor()
    _symptoms = data_preprocessor.extract_symptoms(_prompt)
    print(_symptoms)

    # initialize roberta model
    torch.cuda.empty_cache()
    model = CustomRobertaForSequenceClassification(num_labels=27869).to(device)
    model.load_model(ROOT_DIR)

    # get model
    predictions = model.predict(_symptoms)
    print(predictions)
    predictions = list(predictions[0])

    # free up memory
    del model
    torch.cuda.empty_cache()

    # map to paper ids
    list_of_articles = get_list_of_articles(predictions)
    # summarize papers
    # to reduce memory ussage, split into chunks of 2
    list_of_articles = [[article[:len(article)//2],article[len(article)//2:]] for article in list_of_articles]

    _halves_summarized = []

    # summarize halves individually
    for article in list_of_articles:
        for halve in article:
            _prompt = data_preprocessor.summarize_text_with_format(halve)
            _halves_summarized.append(generate_response(_prompt, generator, len(_prompt)))

    _halves_summarized = [halve[141:] for halve in _halves_summarized]
    print(_halves_summarized)

    _response = ""
    for i in range(0,len(_halves_summarized),2):
        _response += _halves_summarized[i] + _halves_summarized[i+1] + "  \n  \n  \n"

    # type out response word by word
    for word in _response.split(r"\s+"):
        yield word + " "
       # time.sleep(0.02)

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
