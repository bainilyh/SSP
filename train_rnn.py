import torch
import torch.nn as nn
import math
import time
import numpy as np

from models import RNNModel

from torch.utils.data import DataLoader
from sequence_dataset import TextSequenceDataset
from tqdm import tqdm


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
            
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, Timer()
    metric = Accumulator(2)
    desc = '  - (Training)   '
    pre_batch_size = 512
    for batch in tqdm(train_iter, mininterval=2, desc=desc, leave=False):
        X, Y = patch_seq_train2(batch)
        if X.shape[0] != pre_batch_size:
            state = None
        pre_batch_size = X.shape[0]
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * size(y), size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, lr, num_epochs, device,
              use_random_iter=False, prefix_valid=None):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    #                         legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 1, net, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            if prefix_valid is not None:
                print(predict(prefix_valid) + f' ppl:{ppl:.1f}')
            else:
                print(predict(f' ppl:{ppl:.1f}'))
            # animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    if prefix_valid is not None:
        print(predict(prefix_valid))
   
    
    
def predict_ch8(prefix, num_preds, net, device):
    """Generate new characters following the `prefix`.

    Defined in :numref:`sec_rnn_scratch`"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [prefix[0]]
    get_input = lambda: reshape(tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(y)
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ' '.join(map(str, outputs))
    
    
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
            
def patch_seq_train2(item_seq):
    item_seq, pos_items = item_seq[:, :-1], item_seq[:, 1:]
    return item_seq, pos_items

size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
tensor = torch.tensor


if __name__ == '__main__':
    # 模型
    num_hiddens = 64
    # batch_size = 2
    num_steps = 30
    vocab_size = 11
    rnn_layer = nn.RNN(vocab_size, num_hiddens)
    
    # # 测试
    # state = torch.zeros(size=(1, batch_size, num_hiddens))
    # X = torch.rand(size=(num_steps, batch_size, vocab_size))
    # Y, state_new = rnn_layer(X, state)
    # print(Y.shape, state_new.shape) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RNNModel(rnn_layer, vocab_size=vocab_size).to(device)
    
    prefix = [4,7,4,7,7,5,5,4,6,5,7,5,5,6,6,6,7,5,5,5,3,5,5,5,6,5,5,6,4]
    
    output = predict_ch8(prefix, 1, net, device)
    print(output)
    
    # 训练
    num_epochs, lr = 500, 0.1
    # 创建数据集
    dataset_train = TextSequenceDataset(
        file_path='train.txt'
    )
    dataset_valid = TextSequenceDataset(
        file_path='valid.txt'
    )

    # 创建数据加载器
    training_data = DataLoader(
        dataset_train,
        batch_size=512,      # 批量大小
        shuffle=False,       # 是否打乱数据
        num_workers=1       # 使用多少个子进程加载数据
    )
    
    valid_data = DataLoader(
        dataset_valid,
        batch_size=512,      # 批量大小
        shuffle=False,       # 是否打乱数据
        num_workers=1       # 使用多少个子进程加载数据
    )
    
    train_ch8(net, training_data, lr, num_epochs, device, prefix_valid=prefix, use_random_iter=False)