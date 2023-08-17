import sys

from blackbox_loader import load_llama7b, save_llama7b

model = load_llama7b(sys.argv[1])
save_llama7b(model, sys.argv[1], sys.argv[2])

# TODO: load again?