import coremltools as ct
import numpy as np
import torch
import fewlines
import time

import torch.nn as nn

class TestNN(nn.Module):
    def __init__(self, n):
        super(TestNN, self).__init__()
        self.mlp = nn.Linear(n, n)

    def forward(self, x):
        return self.mlp(x)

def to_coreml(torch_model, batch_size, n, compute_units):
    torch_model = torch_model.cpu()
    torch_model.eval()
    sample = torch.rand(batch_size, n)

    traced_model = torch.jit.trace(torch_model, sample)
    return ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=sample.shape)],
        compute_units=compute_units
    )

if __name__ == '__main__':
    step = 32
    timeout = 4.0
    for log_n in range(8, 15):
        for log_batch in range(5, 14):
            n = 2 ** log_n
            batch_size = 2 ** log_batch
            model = TestNN(n)
            cml_model = to_coreml(model, batch_size, n, ct.ComputeUnit.CPU_AND_NE)
            sample = {'x': np.random.rand(batch_size, n)}

            fmas_iter = batch_size * n * n

            iterations = 0

            start = time.time()
            while True:
                for i in range(step):
                    cml_model.predict(sample)
                duration = time.time() - start
                iterations += step
                if duration > timeout:
                    break
                else:
                    step *= 2

            print(fmas_iter, iterations)

            tops = 2.0 * fmas_iter * iterations / 1e12
            tops_s = tops / duration
            print(f'{n},{batch_size}:{tops_s:.3g} tops, {duration:.3g} seconds total')