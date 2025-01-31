import torch.cuda
from torch.cuda.amp import autocast

def generate_response(_prompt, _generator, num_of_articles: int):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        with autocast():
            response = _generator(
                _prompt,
                do_sample = False,
                temperature = 0.3,
                top_p = 0.9,
                max_new_tokens = 200,
            )

    return response[0]['generated_text']
