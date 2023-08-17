# this is an integration test which compares blackbox mode vs regular torch execution
# we run forward/backward pass and compare outputs and weights after one optimizer step.

import time
import torch
import sys

from blackbox_loader import load_llama7b
import backprop_service
from utils import peak_rss

batch_size = 1
length = 50
test_data_dim = 64
seed = 123001
dropout = 0.1
    
# insane LR to see difference after 1 iteration with 1 sample
lr = 100.0

model_path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'data'

def blackbox_backwards(device='cpu'):
    print(f'main peak rss: {peak_rss()}')
    
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = load_llama7b(model_path, dropout=dropout).to(device)
    print(f'loaded model in {time.time() - start} seconds')

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()

    torch.random.manual_seed(seed)
    logits = model(X, Y)

    print(f'forward pass in {time.time() - start} seconds')
    print(f'main peak rss: {peak_rss()}')
    layer_13 = model.layers[13].loaded_inner()
    weight_before = layer_13.attention.wq.weight[:test_data_dim, :test_data_dim].clone()
    emb_before = model.tok_embeddings.loaded_inner().weight[:test_data_dim, :test_data_dim].clone()

    start = time.time()
    opt.zero_grad()
    loss = model.last_loss
    loss.backward()
    opt.step()
    print(f'backward pass in {time.time() - start} seconds')
    print(f'main peak rss: {peak_rss()}')

    layer_13 = model.layers[13].loaded_inner()
    weight_after = layer_13.attention.wq.weight[:test_data_dim, :test_data_dim].clone()
    emb_after = model.tok_embeddings.loaded_inner().weight[:test_data_dim, :test_data_dim].clone()
    return weight_before, weight_after, logits[0, :length, :test_data_dim].clone(), emb_before, emb_after

def plain_backwards(device='cpu'):
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

def reference_compare(save_test_data=True):
    print('Running blackbox on CPU')
    wb, wa, y, emb_before, emb_after = blackbox_backwards('cpu')

    print('Running plain on CPU')
    wb_plain, wa_plain, y_plain, emb_before_plain, emb_after_plain  = plain_backwards()
    if save_test_data:
        torch.save(wb_plain, 'split_model/test_data/sample_weights_before.pt')
        torch.save(wa_plain, 'split_model/test_data/sample_weights_after.pt')
        torch.save(emb_before_plain, 'split_model/test_data/sample_emb_before.pt')
        torch.save(emb_after_plain, 'split_model/test_data/sample_emb_after.pt')
        torch.save(y_plain, 'split_model/test_data/logits.pt')

    same_before = torch.allclose(wb.cpu(), wb_plain.cpu())
    same_after = torch.allclose(wa.cpu(), wa_plain.cpu())
    same_y = torch.allclose(y.cpu(), y_plain.cpu())
    same_emb_before = torch.allclose(emb_before.cpu(), emb_before_plain.cpu())
    same_emb_after = torch.allclose(emb_after.cpu(), emb_after_plain.cpu())
    txt = lambda ok: '[ OK ]' if ok else '[FAIL]'

    print(f'{txt(same_before)} weights before')
    print(f'{txt(same_after)} weights after')
    print(f'{txt(same_emb_before)} emb weights before')
    print(f'{txt(same_emb_after)} emb weights after')
    print(f'{txt(same_y)} out logits')

def test_data_compare():
    print('Running on CPU')
    wb, wa, y, emb_before, emb_after = blackbox_backwards('cpu')

    wb_plain = torch.load('split_model/test_data/sample_weights_before.pt')
    wa_plain = torch.load('split_model/test_data/sample_weights_after.pt')
    y_plain = torch.load('split_model/test_data/logits.pt')
    emb_before_plain = torch.load('split_model/test_data/sample_emb_before.pt')
    emb_after_plain = torch.load('split_model/test_data/sample_emb_after.pt')

    same_before = torch.allclose(wb.cpu(), wb_plain.cpu())
    same_after = torch.allclose(wa.cpu(), wa_plain.cpu())
    same_y = torch.allclose(y.cpu(), y_plain.cpu())
    same_emb_before = torch.allclose(emb_before.cpu(), emb_before_plain.cpu())
    same_emb_after = torch.allclose(emb_after.cpu(), emb_after_plain.cpu())
    txt = lambda ok: '[ OK ]' if ok else '[FAIL]'

    print(f'{txt(same_before)} weights before')
    print(f'{txt(same_after)} weights after')
    print(f'{txt(same_emb_before)} emb weights before')
    print(f'{txt(same_emb_after)} emb weights after')
    print(f'{txt(same_y)} out logits')

if __name__ == '__main__':
    backprop_service.lr = lr
    if mode == 'data':
        test_data_compare()
    else:
        reference_compare()


"""
python split_model/backward_cmp.py ../llama-2-7b
Running on CPU
Loaded 292 module metadata
Created blank model
processing transformer blocks ................................ DONE
populated all weights to model
loaded model in 89.64134502410889 seconds
forward pass in 33.529903173446655 seconds
backward pass in 129.32752394676208 seconds
Running plain on CPU
loaded plain model in 96.33211588859558 seconds
torch.Size([32, 50, 32000]) torch.Size([1600, 32000]) torch.Size([1600])
plain forward pass in 151.54023575782776 seconds
plain backward pass in 210.3796899318695 seconds
[ OK ] weights before
[ OK ] weights after
[ OK ] out logits

"""
