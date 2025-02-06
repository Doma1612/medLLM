# Deployment

make sure Scispacy medical term extraction model is installed:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```
## 1) Log into huggingface-cli
```bash
huggingface-cli login
```
## 2) Run
run in terminal:
```bash
streamlit run ./deployment/main.py
```
set path if necessary
```bash
export PYTHONPATH="/home/user/path_to_project/medLLM"
```
