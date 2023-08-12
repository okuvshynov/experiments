import sys

from phantom_loader import llama7b_phantom, save_model

model = llama7b_phantom(sys.argv[1])
save_model(model, sys.argv[1], sys.argv[2])