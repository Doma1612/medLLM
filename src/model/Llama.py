import torch
from transformers import pipeline

def generate_response(_prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    generator = pipeline(model=model_name, device=device, torch_dtype=torch.bfloat16)

    response = generator(
        _prompt,
        do_sample = False,
        temperature = 1.0,
        top_p = 1,
        max_new_tokens = 50,
    )
    return response[0]['generated_text']