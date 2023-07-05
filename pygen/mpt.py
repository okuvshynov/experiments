import time
import torch
import transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-30b')

name = 'mosaicml/mpt-30b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)

template = """
This program illustrates {topic} functionality. It is written in python, and reads no input from stdin, command-line arguments, files or network. All inputs are hardcoded to the program. The result of the program is printed to stdout.

{imports}
"""

topics = [
    ('numpy reshape', 'import numpy'),
]

for batch_size_l in range(10):
    start = time.time()
    batch_size = 2 ** batch_size_l
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for t, imps in topics:
            prompt = template.format(topic=t, imports=imps)
            inputs = tokenizer([prompt for _ in range(batch_size)], return_tensors="pt").to('cuda')
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for out in decoded:
                with open(f"out{batch_size}.py", 'a') as f:
                    f.write(out)
    dur = time.time() - start
    print(f'batch_size = {batch_size}, total time = {dur:.3g}')