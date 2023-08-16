import torch
from multiprocessing import Process, Pipe

from utils import intermediate_path, restore_rng_state

lr = 100.0

def process_input(ids):
    device, module_id, input_id, grad_output, freqs_cos, freqs_sin, rng_state = ids

    module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
    input = torch.load(intermediate_path(input_id), map_location=torch.device(device))
    input.requires_grad = True

    opt = torch.optim.SGD(module.parameters(), lr=lr)
    opt.zero_grad()

    restore_rng_state(rng_state, device)

    output = module(input, freqs_cos.to(device), freqs_sin.to(device))
    output.backward(grad_output.to(device))
    opt.step()

    torch.save(module, intermediate_path(module_id))
    return input.grad.to('cpu')

def backprop_service(pipe):
    while True:
        message = pipe.recv()
        pipe.send(process_input(message))

class Backprop:
    def __init__(self):
        self.conn, child_conn = Pipe()
        p = Process(target=backprop_service, args=(child_conn,), daemon=True)
        p.start()

    def run(self, args):
        self.conn.send(args)
        return self.conn.recv()
