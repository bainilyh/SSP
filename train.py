from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp


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


def cal_performance_train2(pred, gold, pad_idx):
    gold_shape = gold.shape
    gold = gold.contiguous().view(-1)
    pred_shape = pred.shape
    pred = pred.view(-1, pred.size(2))
    loss = calculate_loss(pred, gold, pad_idx)
    pred = pred.view(pred_shape)
    gold = gold.view(gold_shape)
    
    pred = pred[:, -1, :].contiguous()
    pred = pred.view(-1, pred.size(1))
    gold = gold[:, -1].contiguous().view(-1)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_performance_train3(step_map, pred, gold, pad_idx):
    gold_shape = gold.shape
    gold = gold.contiguous().view(-1)
    pred_shape = pred.shape
    pred = pred.view(-1, pred.size(2))
    loss = calculate_loss(pred, gold, pad_idx)
    pred = pred.view(pred_shape)
    gold = gold.view(gold_shape)
    
    
    # n = pred.size(1)
    # for i in range(n):
    #     sub_pred = pred[:, i, :].contiguous()
    #     sub_pred = sub_pred.view(-1, sub_pred.size(1))
    #     sub_gold = gold[:, i].contiguous().view(-1)

    #     sub_pred = sub_pred.max(1)[1]
    #     sub_gold = sub_gold.contiguous().view(-1)
    #     non_pad_mask = sub_gold.ne(pad_idx)
    #     n_correct = sub_pred.eq(sub_gold).masked_select(non_pad_mask).sum().item()
    #     n_word = non_pad_mask.sum().item()
    #     step_map[i].append(n_correct)
    #     step_map[i].append(n_word)
        
    # Get predictions for all steps at once
    pred_indices = pred.max(2)[1]  # batch_size x seq_len
    gold = gold.contiguous()  # batch_size x seq_len
    
    # Create mask for non-padding tokens
    non_pad_mask = gold.ne(pad_idx)  # batch_size x seq_len
    
    # Calculate correct predictions
    correct = pred_indices.eq(gold) & non_pad_mask  # batch_size x seq_len
    
    # Sum over batch dimension
    n_correct = correct.sum(dim=0)  # seq_len
    n_word = non_pad_mask.sum(dim=0)  # seq_len
    
    # Update step_map
    for i in range(pred.size(1)):
        step_map[i].append(n_correct[i].item())
        step_map[i].append(n_word[i].item())

    return loss


def cal_performance_valid(pred, gold, pad_idx):
    pred = pred[:, -1, :].contiguous()
    pred = pred.view(-1, pred.size(1))
    

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

def patch_seq_train2(item_seq):
    item_seq, pos_items = item_seq[:, :-1], item_seq[:, 1:]
    return item_seq, pos_items


def patch_seq_valid(item_seq):
    item_seq, pos_items = item_seq[:, :-1], item_seq[:, -1].contiguous().view(-1)
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


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========= Loading Dataset =========#

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
        shuffle=True,       # 是否打乱数据
        num_workers=1       # 使用多少个子进程加载数据
    )
    
    valid_data = DataLoader(
        dataset_valid,
        batch_size=512,      # 批量大小
        shuffle=True,       # 是否打乱数据
        num_workers=1       # 使用多少个子进程加载数据
    )


    model = SSP(n_layers=2, n_heads=4, n_items=11, hidden_size=32, d_inner=128, d_k=8, d_v=8, dropout=0.1, pad_idx=0, max_seq_len=30).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    fit(model, training_data, valid_data, optimizer, device)


if __name__ == '__main__':
    main()
