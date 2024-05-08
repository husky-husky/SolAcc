# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/5/19
import math

import torch

from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import EncoderDecoderModel
from torch.nn import init
import numpy as np

import os

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



class BiLSTM_part(nn.Module):
    """
    hidden1=256, hidden2=512, hidden3=1024, hidden4=1024
    """

    def __init__(self, input_size, hidden1=512, hidden2=256, hidden3=128, hidden4=1024, dropout=0.2):
        super(BiLSTM_part, self).__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.dropout = dropout

        self.layerNorm = nn.LayerNorm(self.input_size)

        self.BiLSTM1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden1, batch_first=True,
                               bidirectional=True, dropout=self.dropout)
        self.BiLSTM2 = nn.LSTM(input_size=self.hidden1 * 2, hidden_size=self.hidden2, batch_first=True,
                               bidirectional=True, dropout=self.dropout)
        self.BiLSTM3 = nn.LSTM(input_size=self.hidden2, hidden_size=self.hidden3, batch_first=True,
                               dropout=self.dropout, bidirectional=True)
        self.BiLSTM4 = nn.LSTM(input_size=self.hidden3, hidden_size=1024, batch_first=True,
                               dropout=self.dropout, bidirectional=True)

        self.BiLSTM5 = nn.LSTM(input_size=self.hidden4, hidden_size=1024, batch_first=True,
                               dropout=self.dropout, bidirectional=True)

    def forward(self, input_features):
        x = self.layerNorm(input_features)

        lstm1, _ = self.BiLSTM1(x)  # 经过第一层双向LSTM，操作是concat，默认输出就是拼接，不做任何操作
        lstm2, _ = self.BiLSTM2(lstm1)  # 经过第二层双向LSTM，操作是sum，需要分割数据

        lstm2_p1, lstm2_p2 = lstm2[:, :, 0:self.hidden2], lstm2[:, :, self.hidden2:]
        lstm2 = lstm2_p1 + lstm2_p2

        lstm3, _ = self.BiLSTM3(lstm2)  # 经过第三层lstm，操作是weighting sum，同样需要分割数据

        lstm3_p1, lstm3_p2 = lstm3[:, :, 0:self.hidden3], lstm3[:, :, self.hidden3:]

        lstm3 = lstm3_p1 + lstm3_p2

        # lstm4, _ = self.BiLSTM4(lstm3)
        # lstm4_p1, lstm4_p2 = lstm4[:, :, 0:self.hidden4], lstm4[:, :, self.hidden4:]
        # lstm4 = lstm4_p1 + lstm4_p2

        # lstm5, _ = self.BiLSTM5(lstm4)
        # lstm5_p1, lstm5_p2 = lstm5[:, :, 0:self.hidden4], lstm5[:, :, self.hidden4:]
        # lstm5 = lstm5_p1 + lstm5_p2

        return lstm3

    def predict(self, input_features):
        return self.forward(input_features)


class TransformerNet(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim=128, num_heads=4, num_layers=6, dropout_rate=.2):
        super(TransformerNet, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.input_linear = nn.Linear(50, 128)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                              dim_feedforward=embedding_dim * 4, dropout=dropout_rate)
        self.transformer_encoder_layers = nn.TransformerEncoder(self.transformer_encoder, num_layers=num_layers)

        self.output_linear = nn.Linear(embedding_dim, output_size)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x, attention_mask=None):
        encoded_input = self.relu(self.input_linear(self.layer_norm(x)))
        encoded_output = self.transformer_encoder_layers(encoded_input)
        if attention_mask is not None:
            # 取出有效部分
            embedding = []
            # 对lstm的输出进行降维到两维
            for seq_num in range(len(attention_mask)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = encoded_output[seq_num][0: seq_len]
                embedding.append(seq_emd)
            pooled_output = torch.cat(embedding, 0)
            output = self.output(pooled_output)
            return output
        return encoded_output


class Transformer_BiLSTM(nn.Module):
    def __init__(self, input_size):
        super(Transformer_BiLSTM, self).__init__()
        self.transformer_part = TransformerNet(input_size, 128)
        self.BiLSTM_part = BiLSTM_part(input_size)
        self.attention_layer = DotProductAttention(.1)
        self.output_linear = nn.Linear(128, 1)
        self.linear = nn.Linear(256, 128)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input_features, attention_mask):

        lstm_output = self.BiLSTM_part(input_features)
        transformer_output = self.transformer_part(input_features)

        # attention_output = lstm_output + transformer_output
        # attention_output = self.relu(self.linear(torch.cat((lstm_output, transformer_output), dim=-1)))

        #  点积融合
        valid_len = []
        for seq_num in range(len(attention_mask)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            valid_len.append(seq_len)
        valid_len = torch.tensor(valid_len)

        # add_concat = lstm_output + transformer_output

        attention_output = self.attention_layer(lstm_output, transformer_output, transformer_output, valid_len)
        # ## 加性融合
        # attention_output = lstm_output + transformer_output

        # # concat融合
        # attention_output = torch.cat((lstm_output, transformer_output), dim=-1)

        # flatten
        embedding = []
        for seq_num in range(len(attention_mask)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = attention_output[seq_num][0: seq_len]
            embedding.append(seq_emd)
        embedding = torch.cat(embedding, 0)
        output = self.sigmoid(self.output_linear(embedding))
        return output

    def predict(self, input_features, attention_mask):
        return self.forward(
            input_features=input_features,
            attention_mask=attention_mask
        )



class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(attention_weights), values)


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


if __name__ == "__main__":
    pass
   
