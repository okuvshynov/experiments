import torch
import torch.multiprocessing as mp

from utils import intermediate_path, restore_rng_state, peak_rss

def process_input(args):
    lr, device, module_id, input_id, grad_output, rng_state, *extra = args

    module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
    input = torch.load(intermediate_path(input_id), map_location=torch.device(device))
    
    # no grad for embedding inputs
    # TODO: better check here
    if 'float' in str(input.dtype):
        input.requires_grad = True

    opt = torch.optim.SGD(module.parameters(), lr=lr)
    opt.zero_grad()

    restore_rng_state(rng_state, device)

    extra = [t.to(device) for t in extra]

    output = module(input, *extra)
    output.backward(grad_output.to(device))
    opt.step()

    torch.save(module, intermediate_path(module_id))
    print(f'learner peak rss: {peak_rss()}')
    return input.grad.to('cpu') if input.requires_grad else None

def backprop_service(pipe):
    while True:
        message = pipe.recv()
        pipe.send(process_input(message))

lr = 1e-5

class Backprop:
    def __init__(self):
        mp.set_start_method('spawn')
        self.conn, child_conn = mp.Pipe()
        p = mp.Process(target=backprop_service, args=(child_conn,), daemon=True)
        p.start()

    def run(self, args):
        self.conn.send([lr] + args)
        return self.conn.recv()
