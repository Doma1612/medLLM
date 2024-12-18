from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    # TODO: get correct model name
    model_name = "huggingface/llama-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)
