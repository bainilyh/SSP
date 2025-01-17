# 使用timemachine数据训练ssp模型
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp
from d2l import torch as d2l


from tqdm import tqdm

import os
import time
import math

from models import SSP
from sequence_dataset import TextSequenceDataset

def calculate_loss(seq_output, pos_items, pad_idx):
    # seq_output = model(item_seq) # batch_size * seq, n_items
    # pos_items = pos_items.view(-1, 1) # batch_size, 1
    loss = F.cross_entropy(seq_output, pos_items,
                           ignore_index=pad_idx, reduction='sum')
    return loss


def cal_performance(pred, gold, pad_idx):

    loss = calculate_loss(pred, gold, pad_idx)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def patch_seq(item_seq):
    item_seq, pos_items = item_seq[:, :-1], item_seq[:, 1:].contiguous().view(-1)
    return item_seq, pos_items



def train_epoch(model, training_data, optimizer, device):
    model.train()

    total_loss = 0        # 总损失
    n_word_total = 0      # 总单词数
    n_word_correct = 0    # 预测正确的单词数
    
    max_seq_len = model.max_seq_len
    # total_loss = 0
    # n_word_total = [0] * max_seq_len
    # n_word_correct = [0] * max_seq_len
    

    # 使用tqdm显示训练进度条
    desc = '  - (Training)   '
    scaler = amp.GradScaler(enabled=True)
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        step_map = defaultdict(list)
        # 准备数据：将源序列和目标序列移到指定设备(CPU/GPU)
        # trg_seq用于输入，gold用于计算损失
        item_seq, gold = map(lambda x: x.to(device), patch_seq(batch))

        # 清零梯度
        optimizer.zero_grad()
        # 前向传播：通过模型获取预测结果
        pred = model(item_seq)
        # 计算损失和准确率
        loss, n_correct, n_word = cal_performance(pred, gold, 0)
        # loss = cal_performance_train3(step_map, pred, gold, 0)
        # 反向传播计算梯度
        scaler.scale(loss).backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        # 累计统计数据
        n_word_total += n_word            # 累计单词总数
        n_word_correct += n_correct       # 累计正确预测数
        total_loss += loss.item()         # 累计损失值
        # for i in range(max_seq_len):
        #     n_word_total[i] += step_map[i][1]
        #     n_word_correct[i] += step_map[i][0]
        # total_loss += loss.item()    
        # for name, param in model.named_parameters():
        #     if param.grad is not None and name == 'item_embedding.weight':
        #         grad_norm = param.grad.norm().item()  # 将张量转换为Python标量
        #         grad_norm_list.append(grad_norm)
    
    # print(f"Gradient norm for {name}: {grad_norm:.5f}".format(name='item_embedding.weight', grad_norm=sum(grad_norm_list)/len(grad_norm_list)))
            
    # 计算整个epoch的平均损失和准确率
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    # # 计算整个epoch的平均损失和准确率
    # loss_per_word = total_loss/sum(n_word_total)
    # accuracy = list()
    # for i in range(max_seq_len):
    #     accuracy.append(n_word_correct[i]/n_word_total[i])
    # return loss_per_word, accuracy



def eval_epoch(model, validation_data, device):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            item_seq, gold = map(lambda x: x.to(device), patch_seq_valid(batch))

            # forward
            pred = model(item_seq, need_reshape=False)
            loss, n_correct, n_word = cal_performance_valid(pred, gold, 0)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def fit(model, training_data, valid_data, optimizer, device):

    log_train_file = os.path.join('train.log')
    log_valid_file = os.path.join('valid.log')

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, ppl, accu, start_time, lr):
        if isinstance(accu, list):
            print('  - {header:12} loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu}, lr: {lr:8.5f}, '
                    'elapse: {elapse:3.3f} min'.format(
                        header=f"({header})", loss=loss, ppl=ppl,
                        accu=[str(a * 100) + '%' for a in accu], elapse=(time.time()-start_time)/60, lr=lr))
        else:
            print('  - {header:12} loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '
                    'elapse: {elapse:3.3f} min'.format(
                        header=f"({header})", loss=loss, ppl=ppl,
                        accu=accu * 100, elapse=(time.time()-start_time)/60, lr=lr))

    valid_accus = []
    valid_losses = []
    for epoch_i in range(3000):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer.param_groups[0]['lr']
        print_performances('Training', train_loss, train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, valid_data, device)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_loss, valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i,
                      'model': model.state_dict()}

        model_name = 'ssp.chkpt'
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, os.path.join('./model', model_name))
        print('    - [Info] The checkpoint file has been updated.')

        # with open(log_train_file, 'a') as log_tf:
        #     log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
        #         epoch=epoch_i, loss=train_loss,
        #         ppl=train_ppl, accu=100*train_accu))



def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()

    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.Adam(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: d2l.predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            # print(predict('time traveller') + f' perplexity {ppl:.1f}')
            print(f' perplexity {ppl:.1f}')
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))
    
    
# def predict_ch8(prefix, num_preds, net, vocab, device):
#     """Generate new characters following the `prefix`.

#     Defined in :numref:`sec_rnn_scratch`"""
#     for _ in range(num_preds):  # Predict `num_preds` steps
#         y, state = net(get_input(), state)
#         outputs.append(int(y.argmax(dim=1).reshape(1)))
#     return ''.join([vocab.idx_to_token[i] for i in outputs])

    
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        y = Y.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            d2l.grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            d2l.grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    
# def generate(model, prefix_seq, num_preds):
#     model.eval()
#     text = predict('time traveller')
#     print(text)
    
    
    
def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========= Loading Dataset =========#
    batch_size, num_steps = 512, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps, max_tokens=-1)

    model = SSP(n_layers=3, n_heads=4, n_items=len(vocab), hidden_size=32, d_inner=128, d_k=8, d_v=8, dropout=0.1, pad_idx=0, max_seq_len=35).to(device)

    num_epochs, lr = 500, 0.01
    
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)


if __name__ == '__main__':
    main()
