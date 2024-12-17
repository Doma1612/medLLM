# Instructions: Accessing LLMs via Hugging Face

This guide explains how to request access to the **Llama 2-7B** and **Mistral 7B** models on Hugging Face and configure your environment to use these models within the `medLLM` repository.

## Step 1: Request Model Access on Hugging Face
1. Visit the following links and request access:
   - [Llama 2-7B](https://huggingface.co/meta-llama/Llama-2-7b)
   - [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
2. For **Mistral 7B**, access is granted immediately.
3. For **Llama 2-7B**, your request will be forwarded to Meta for approval. This process may take a few minutes.

## Step 2: Create a Hugging Face Access Token
1. Log in to Hugging Face and click on your profile in the top-right corner.
2. Navigate to **Access Tokens** under your profile menu.
3. Click **Create New Token**:
   - **Token Name**: Choose a descriptive name.
   - **Role**: Select `Write` as the token type.
4. Save the token securely.

## Step 3: Save the Token in the `medLLM` Repository
1. Navigate to the `src/config` directory in the `medLLM` repository.
2. Create or open the `secrets.py` file.
3. Add the following line, replacing `YOUR_TOKEN` with your Hugging Face token:
   ```python
   huggingface_token = "YOUR_TOKEN"
The secrets.py file is ignored by Git to keep the token secure.

## Step 4: Access the Models in a Notebook

Install the required Hugging Face package if not already installed:
```bash
pip install huggingface_hub
```
In your notebook, log in to Hugging Face using the following code:
```python
from huggingface_hub import notebook_login
notebook_login(write_permission=True)
```

A window will open where you can paste your Hugging Face token. After authentication, you'll have access to the models.

## Step 5: Verify Access
Open the notebooks in the repository to see how the models are accessed and ensure everything is configured correctly.
Ensure you have the necessary permissions to use these models in your environment.
Do not share your access token or commit it to version control.
For any issues, refer to the Hugging Face documentation or contact the repository maintainers.


## Important:
If your notebook does not support GPU, please consider using google colab.