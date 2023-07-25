# teaching model to do addition

out_dir = 'out-addition'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 25 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'addition'
wandb_run_name = 'mini-gpt'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 64 # context of up to 64 previous characters

# even smaller model
n_layer = 2
n_head = 6
n_embd = 96

# model is even smaller, can make dropout rate smaller as well
dropout = 0.1

# slightly higher LR
learning_rate = 2e-3 
max_iters = 20000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model