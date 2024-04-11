# check if some parts of transformer model can be offloaded to ANE

import torch.nn as nn
import coremltools as ct
import numpy as np
import time
import torch

in_dim = 4096

class FeedForward(nn.Module):
    def __init__(self, dim=in_dim, hidden_dim=14336):
        super().__init__()

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


def to_coreml(torch_model, batch_size, compute_units):
    torch_model = torch_model.cpu()
    torch_model.eval()
    sample = torch.rand(batch_size, in_dim)

    traced_model = torch.jit.trace(torch_model, sample)
    return ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)],
        compute_units=compute_units
    )

def try_gpu(batch_sizes=[16, 32, 1]):
    model = FeedForward().to('mps').to(torch.float16)
    run_for_nseconds = 30
    step_target = 0.05 * run_for_nseconds
    for batch_size in batch_sizes:
        sample = torch.rand(batch_size, in_dim).to('mps').to(torch.float16)

        start = time.time()
        it = 0
        step = 100
        while True:
            for _ in range(step):
                out = model(sample)
            it += step
            curr = time.time()
            if curr > run_for_nseconds + start:
                break
            if curr < start + step_target:
                step *= 2

        duration = time.time() - start
        total_ranked = it * batch_size
        ms_per_sample = 1000.0 * duration / total_ranked

        print(f'{batch_size},{duration:.3f},{total_ranked},{ms_per_sample:.3f}')
  
def try_ane(batch_sizes=[16, 32, 1]):
    run_for_nseconds = 30
    step_target = 0.05 * run_for_nseconds

    model = FeedForward().to('mps')
    for batch_size in batch_sizes:

        sample = {'x': np.random.rand(batch_size, 4096)}

        ne_model = to_coreml(model, batch_size, compute_units=ct.ComputeUnit.CPU_AND_NE)

        start = time.time()
        it = 0
        step = 100
        while True:
            for _ in range(step):
                out = ne_model.predict(sample)
            it += step
            curr = time.time()
            if curr > run_for_nseconds + start:
                break
            if curr < start + step_target:
                step *= 2

        duration = time.time() - start
        total_ranked = it * batch_size
        ms_per_sample = 1000.0 * duration / total_ranked

        print(f'{batch_size},{duration:.3f},{total_ranked},{ms_per_sample:.3f}')


if __name__ == "__main__":
    try_ane([1024, 2048, 512])
    try_gpu([128, 256, 512])
