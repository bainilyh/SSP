# import torch
# from torch import nn
# from torch.nn import functional as F
# from d2l import torch as d2l

# batch_size, num_steps = 32, 35
# train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps, max_tokens=1000000)

# vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# num_epochs, lr = 500, 1
# num_inputs = vocab_size

# gru_layer = nn.GRU(num_inputs, num_hiddens)
# model = d2l.RNNModel(gru_layer, len(vocab))
# model = model.to(device)


# def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
#               use_random_iter=False):
#     """Train a model (defined in Chapter 8).

#     Defined in :numref:`sec_rnn_scratch`"""
#     loss = nn.CrossEntropyLoss()

#     # Initialize
#     if isinstance(net, nn.Module):
#         updater = torch.optim.SGD(net.parameters(), lr)
#     else:
#         updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
#     predict = lambda prefix: d2l.predict_ch8(prefix, 50, net, vocab, device)
#     # Train and predict
#     for epoch in range(num_epochs):
#         ppl, speed = d2l.train_epoch_ch8(
#             net, train_iter, loss, updater, device, use_random_iter)
#         if (epoch + 1) % 10 == 0:
#             print(predict('time traveller') + f' perplexity {ppl:.1f}')
#     print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
#     print(predict('time traveller'))
#     print(predict('traveller'))
    
# train_ch8(model, train_iter, vocab, lr, num_epochs, device)