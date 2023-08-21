# this is an integration test which compares blackbox mode vs regular torch execution
# we run forward/backward pass and compare outputs and weights after one optimizer step.

import time
import torch
import sys

from blackbox_loader import load_llama7b
from utils import peak_rss_mb

batch_size = 1
length = 50
test_data_dim = 64
seed = 123001
dropout = 0.1
    
# insane LR to see difference after 1 iteration with 1 sample
lr = 100.0

model_path = sys.argv[1]

# ref -- reference implementation. 
# data -- precomputed data. 
# None - no comparison, just run and measure time
compare_to = sys.argv[2] if len(sys.argv) > 2 else 'none'

device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'

test_data_paths = [
    'split_model/test_data/sample_weights_before.pt',
    'split_model/test_data/sample_weights_after.pt',
    'split_model/test_data/logits.pt',
    'split_model/test_data/sample_emb_before.pt',
    'split_model/test_data/sample_emb_after.pt'
]

txt = lambda ok: '[ OK ]' if ok else '[FAIL]'

def blackbox_backwards():
    print(f'peak rss: {peak_rss_mb()}')
    
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = load_llama7b(model_path, dropout=dropout).to(device)
    print(f'loaded model in {time.time() - start} seconds')

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()

    torch.random.manual_seed(seed)
    logits = model(X, Y)

    print(f'forward pass in {time.time() - start} seconds, peak rss {peak_rss_mb()}')
    layer_13 = model.layers[13].loaded_inner()
    weight_before = layer_13.attention.wq.weight[:test_data_dim, :test_data_dim].clone()
    emb_before = model.tok_embeddings.loaded_inner().weight[:test_data_dim, :test_data_dim].clone()

    start = time.time()
    opt.zero_grad()

    torch.random.manual_seed(seed)
    logits2, loss2 = model.manual_loop(X, Y, lr=lr)

    opt.step()
    print(f'combined pass in {time.time() - start} seconds, peak rss {peak_rss_mb()}')

    print(f'{txt(torch.allclose(logits.cpu(), logits2.cpu()))} logits from fwd/combined are same')

    layer_13 = model.layers[13].loaded_inner()
    weight_after = layer_13.attention.wq.weight[:test_data_dim, :test_data_dim].clone()
    emb_after = model.tok_embeddings.loaded_inner().weight[:test_data_dim, :test_data_dim].clone()
    return weight_before, weight_after, logits[0, :length, :test_data_dim].clone(), emb_before, emb_after

def plain_backwards():
    from plain_loader import llama7b_torch
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = llama7b_torch(model_path, dropout=dropout).to(device)
    print(f'loaded plain model in {time.time() - start} seconds')

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()
    torch.random.manual_seed(seed)
    logits = model(X, Y)
    print(f'plain forward pass in {time.time() - start} seconds')
    weight_before = model.layers[13].attention.wq.weight[:test_data_dim, :test_data_dim].clone()
    emb_before = model.tok_embeddings.weight[:test_data_dim, :test_data_dim].clone()

    start = time.time()
    opt.zero_grad()
    loss = model.last_loss
    loss.backward()
    opt.step()
    print(f'plain backward pass in {time.time() - start} seconds')

    weight_after = model.layers[13].attention.wq.weight[:test_data_dim, :test_data_dim].clone()
    emb_after = model.tok_embeddings.weight[:test_data_dim, :test_data_dim].clone()
    return weight_before, weight_after, logits[0, :length, :test_data_dim].clone(), emb_before, emb_after

def get_comparison_data():
    return plain_backwards() if compare_to == 'ref' else tuple(torch.load(p) for p in test_data_paths)

def run(save_test_data):
    wb, wa, y, emb_before, emb_after = blackbox_backwards()
    if compare_to == 'none':
        return
    comparison_data = get_comparison_data()
    if save_test_data:
        for t, p in zip(comparison_data, test_data_paths):
            torch.save(t, p)

    wb_plain, wa_plain, y_plain, emb_before_plain, emb_after_plain = comparison_data
    
    print(f'{txt(torch.allclose(wb.cpu(), wb_plain.cpu()))} weights before')
    print(f'{txt(torch.allclose(wa.cpu(), wa_plain.cpu()))} weights after')
    print(f'{txt(torch.allclose(emb_before.cpu(), emb_before_plain.cpu()))} emb weights before')
    print(f'{txt(torch.allclose(emb_after.cpu(), emb_after_plain.cpu()))} emb weights after')
    print(f'{txt(torch.allclose(y.cpu(), y_plain.cpu()))} out logits')

if __name__ == '__main__':
    run(save_test_data=False)
