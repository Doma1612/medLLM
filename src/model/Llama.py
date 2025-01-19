import torch.cuda
from torch.cuda.amp import autocast

def generate_response(_prompt, _generator):

    with autocast():
        response = _generator(
            _prompt,
            do_sample = False,
            temperature = 1.0,
            top_p = 1,
            max_new_tokens = 20,
        )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return response[0]['summary_text']