from transformers import LlamaForCausalLm, LlamaTokenizer

def load_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name, local_files_only=True)
    model = LlamaForCausalLm.from_pretrained(model_name, local_files_only=True)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)
