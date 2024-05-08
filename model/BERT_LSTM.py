# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/5/19
import math

import torch

from torch import nn
import torch.nn.functional as F
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


class AFT_FULL(nn.Module):

    def __init__(self, d_model, n, simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if simple:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, inputs1, inputs2):

        bs, n, dim = inputs1.shape

        q = self.fc_q(inputs1)  # bs,n,dim
        k = self.fc_k(inputs1).view(1, bs, n, dim)  # 1,bs,n,dim
        v = self.fc_v(inputs2).view(1, bs, n, dim)  # 1,bs,n,dim

        print("after qkv:")

        os.system("nvidia-smi")

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n,bs,dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n,bs,dim
        os.system("nvidia-smi")

        out = (numerator / denominator)  # n,bs,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs,n,dim

        return out




class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(attention_weights), values)




if __name__ == "__main__":
    pass
