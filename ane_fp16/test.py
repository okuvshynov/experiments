import torch

dim = 2 ** 14

torch.manual_seed(123)

a32 = torch.rand((dim, dim)).to('mps').to(torch.float32)
b32 = torch.rand((dim, dim)).to('mps').to(torch.float32)

c32 = a32 @ b32

a16 = torch.rand((dim, dim)).to('mps').to(torch.float16)
b16 = torch.rand((dim, dim)).to('mps').to(torch.float16)

c16 = a16 @ b16

diff = torch.abs(c32 - c16.to(torch.float32)).view(-1).to('cpu')
q = torch.tensor([0.5, 0.95, 0.99])
print(torch.quantile(diff, q))