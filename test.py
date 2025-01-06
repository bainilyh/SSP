# from models import SSP
# from train import *

# import torch

# model = SSP(n_layers=6, n_heads=8, n_items=10000, hidden_size=256, d_inner=512, d_k=64, d_v=64, dropout=0.1, pad_idx=0, max_seq_len=200)

# input_ids = torch.randint(0, 10000, size=(4, 31))
# item_seq, pos_items = patch_seq(item_seq=input_ids)

# output = model(item_seq)

# loss, n_correct, n_word = cal_performance(output, pos_items, 0)

# print(loss, n_correct, n_word)


import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import re

# 数据
batch_size, num_steps = 32, 35


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset.

    Defined in :numref:`sec_language_model`"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab 


class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
    
def load_corpus_time_machine(max_tokens=-1):
    """Return token indices and the vocabulary of the time machine dataset.

    Defined in :numref:`sec_text_preprocessing`"""
    lines = read_time_machine()
    tokens = d2l.tokenize(lines, 'char')
    vocab = d2l.Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def read_time_machine():
    """Load the time machine dataset into a list of text lines.

    Defined in :numref:`sec_text_preprocessing`"""
    with open('train.txt', 'r') as f:
        lines = f.readlines()
    return [line.strip().split(' ') for line in lines]
    
train_iter, vocab = load_data_time_machine(batch_size, num_steps, max_tokens=1000000)

print("数据加载完成！")
# 模型
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)


class RNNModel(nn.Module):
    """循环神经⽹络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层⾸先将Y的形状改为(时间步数*批量⼤⼩,隐藏单元数)
        # 它的输出形状是(时间步数*批量⼤⼩,词表⼤⼩)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,batch_size, self.num_hiddens), device=device),
                    torch.zeros(( self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    #                         legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: d2l.predict_ch8(prefix, 10, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = d2l.train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict(['4', '7', '4', '8', '7', '5', '5', '4', '6', '5', '7', '5', '5', '6', '6', '6', '7', '5', '5', '5']) + f'ppl: {ppl:.1f}')
            # animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict(['4', '7', '4', '8', '7', '5', '5', '4', '6', '5', '7', '5', '5', '6', '6', '6', '7', '5', '5', '5']))
    # print(predict('traveller'))
    
# 测试
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8(['4', '7', '4', '8', '7', '5', '5', '4', '6', '5', '7', '5', '5', '6', '6', '6', '7', '5', '5', '5'], 10, net, vocab, device)

# 训练
num_epochs, lr = 500, 0.1
train_ch8(net, train_iter, vocab, lr, num_epochs, device)
