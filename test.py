from models import SSP
from train import *

import torch

model = SSP(n_layers=6, n_heads=8, n_items=10000, hidden_size=256, d_inner=512, d_k=64, d_v=64, dropout=0.1, pad_idx=0, max_seq_len=200)

input_ids = torch.randint(0, 10000, size=(4, 31))
item_seq, pos_items = patch_seq(item_seq=input_ids)

output = model(item_seq)

loss, n_correct, n_word = cal_performance(output, pos_items, 0)

print(loss, n_correct, n_word)

