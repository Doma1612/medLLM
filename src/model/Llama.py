import torch
import torch.cuda
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from transformers import pipeline

class LlamaModel:
    def __init__(self, device):
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.generator = pipeline(model=self.model_name, device=device, torch_dtype=torch.float16)

    def generate_response(self, _prompt):

        input_tokens = len(self.tokenizer(_prompt))
        max_limit = self.tokenizer.model_max_length

        max_new_token = min(500, max_limit - input_tokens)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        with torch.no_grad():
            with autocast():
                response = self.generator(
                    _prompt,
                    do_sample = True,
                    temperature = 0.3,
                    top_p = 0.9,
                    max_new_tokens = max_new_token,
                )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return response[0]['generated_text']
