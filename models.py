import torch
import torch.nn as nn
import torch.nn.functional as F


class SSP(nn.Module):
    def __init__(self, n_layers, n_heads,
                 n_items, hidden_size, d_inner, d_k, d_v,
                 dropout=0.1, pad_idx=0, max_seq_len=200):
        super(SSP, self).__init__()

        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(
            n_items, hidden_size, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.transformer_decoder = TransformerDecoder(
            n_layers, n_heads, d_k, d_v, hidden_size, d_inner, dropout)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        self.item_prj = nn.Linear(hidden_size, n_items, bias=False)
        self.item_prj.weight = self.item_embedding.weight
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.apply(self._init_weights)

    def forward(self, item_seq, need_reshape=True):
        # 1. 位置embedding
        position_ids = torch.arange(item_seq.size(
            1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # 2. 物品embedding
        item_embedding = self.item_embedding(item_seq)

        # 3. 融合位置embedding和物品embedding
        hidden_states = item_embedding + position_embedding
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # 4. 生成注意力掩码
        attention_mask = self.get_attention_mask(item_seq)
        
        # 5. 解码器
        #  decoder_output是一个list，包含n_layers个decoder_layer的输出
        decoder_output = self.transformer_decoder(hidden_states, attention_mask)
        output = decoder_output[-1]
        
        # # 6. 收集最后一个时间步的输出
        # output = self.gather_indexes(output, item_seq_len - 1)
        
        # 7. 投影到item空间
        output = self.item_prj(output)
        if not need_reshape:
            return output

        return output.view(-1, output.size(2)) # batch_size, n_items

    def predict(self, x):
        pass


    def print_model_parameters(self):
        print(self.item_embedding.weight.data[16])
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = torch.tril(
            extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
        )
        extended_attention_mask = torch.where(
            extended_attention_mask, 0.0, -1e9)
        return extended_attention_mask
    
    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer_stack:
            hidden_states = layer_module(hidden_states, attention_mask)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, hidden_states, attention_mask=None):
        dec_output = self.slf_attn(
            hidden_states, hidden_states, hidden_states, mask=attention_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v)
        

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        q = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask != 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

