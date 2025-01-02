import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SSP


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SSP(n_layers=2, n_heads=2, n_items=21, hidden_size=16, d_inner=64, d_k=8, d_v=8, dropout=0.1, pad_idx=0, max_seq_len=30).to(device)
    checkpoint = torch.load('ssp.chkpt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    item_seq = torch.tensor([[9, 16, 10, 12, 10, 14, 6, 5, 15, 14, 13, 11, 15, 15, 17, 10, 10, 19, 6, 6, 10, 14, 7, 18, 4, 11, 14, 7, 4, 10]]).to(device)
    pred = model(item_seq, need_reshape=False)
    pred = pred.max(1)[1]
    print(pred)

