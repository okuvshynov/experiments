import time
from multiprocessing import Process, Pipe

import torch
import sys

from utils import intermediate_path, restore_rng_state


lr = 100.0

def process_input(ids):
    device, module_id, input_id, grad_output_id, grad_input_id, freqs_cos_id, freqs_sin_id, rng_state_id = ids

    module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
    input = torch.load(intermediate_path(input_id), map_location=torch.device(device))
    freqs_cos = torch.load(intermediate_path(freqs_cos_id), map_location=torch.device(device))
    freqs_sin = torch.load(intermediate_path(freqs_sin_id), map_location=torch.device(device))
    input.requires_grad = True

    opt = torch.optim.SGD(module.parameters(), lr=lr)
    opt.zero_grad()

    grad_output = torch.load(intermediate_path(grad_output_id), map_location=torch.device(device))

    rng_state = torch.load(intermediate_path(rng_state_id))
    restore_rng_state(rng_state, device)

    output = module(input, freqs_cos, freqs_sin)
    output.backward(grad_output)
    opt.step()

    torch.save(input.grad, intermediate_path(grad_input_id))
    torch.save(module, intermediate_path(module_id))

def backprop_service(pipe):
    while True:
        message = pipe.recv()
        print(message)
        process_input(message)
        pipe.send('Y')


class Backprop:
    def __init__(self):
        self.parent_conn, child_conn = Pipe()
        p = Process(target=backprop_service, args=(child_conn,))
        p.start()

    def run(self, args):
        self.parent_conn.send(args)
        resp = self.parent_conn.recv()
        return