import torch.cuda
from torch.cuda.amp import autocast

def generate_response(_prompt, _generator, num_of_articles: int):

    with autocast():
        response = _generator(
            _prompt,
            do_sample = False,
            temperature = 1.0,
            top_p = 1,
            max_new_tokens = num_of_articles*200,
        )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return response[0]['summary_text']