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


def get_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    BERT_model = BertModel.from_pretrained(model_path)
    return tokenizer, BERT_model


# @save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        # shape = X.shape
        # if valid_lens.dim() == 1:
        #     valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        # else:
        #     valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        # X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
        #                       value=-1e6)
        for index, length in enumerate(valid_lens):
            X[index][length.tolist():] = -1e6
        return F.softmax(X, dim=-1)


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


class BiLSTMModel(nn.Module):
    """
    hidden1=256, hidden2=512, hidden3=1024, hidden4=1024
    """

    def __init__(self, input_size, hidden1=64, hidden2=128, hidden3=256, hidden4=512, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.dropout = dropout

        self.layerNorm = nn.LayerNorm(self.input_size)

        self.BiLSTM1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden1, batch_first=True,
                               bidirectional=True, dropout=self.dropout)
        self.BiLSTM2 = nn.LSTM(input_size=self.hidden1, hidden_size=self.hidden2, batch_first=True,
                               bidirectional=True, dropout=self.dropout)
        self.BiLSTM3 = nn.LSTM(input_size=self.hidden2, hidden_size=self.hidden3, batch_first=True,
                               dropout=self.dropout, bidirectional=True)
        self.BiLSTM4 = nn.LSTM(input_size=self.hidden3, hidden_size=self.hidden4, batch_first=True,
                               dropout=self.dropout, bidirectional=True)

        self.resnet_layer = nn.Linear(50, 512)
        self.output_linear = nn.Linear(512, 1)
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

    def forward(self, input_features, attention_mask):
        x = self.layerNorm(input_features)

        lstm1, _ = self.BiLSTM1(x)
        lstm1_p1, lstm1_p2 = lstm1[:, :, 0:self.hidden1], lstm1[:, :, self.hidden1:]
        lstm1 = lstm1_p1 + lstm1_p2

        lstm2, _ = self.BiLSTM2(lstm1)  # 经过第二层双向LSTM，操作是sum，需要分割数据
        lstm2_p1, lstm2_p2 = lstm2[:, :, 0:self.hidden2], lstm2[:, :, self.hidden2:]
        lstm2 = lstm2_p1 + lstm2_p2

        lstm3, _ = self.BiLSTM3(lstm2)  # 经过第三层lstm，操作是weighting sum，同样需要分割数据
        lstm3_p1, lstm3_p2 = lstm3[:, :, 0:self.hidden3], lstm3[:, :, self.hidden3:]
        lstm3 = lstm3_p1 + lstm3_p2

        lstm4, _ = self.BiLSTM4(lstm3)
        lstm4_p1, lstm4_p2 = lstm4[:, :, 0:self.hidden4], lstm4[:, :, self.hidden4:]
        lstm4 = lstm4_p1 + lstm4_p2

        # 只与最后一层残差连接
        # res_output = self.resnet_layer(x)
        # lstm4 = lstm4 + res_output

        # flatten
        embedding = []
        for seq_num in range(len(attention_mask)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = lstm4[seq_num][0: seq_len]
            embedding.append(seq_emd)
        embedding = torch.cat(embedding, 0)
        output = self.sigmoid(self.output_linear(embedding))
        return output

    def predict(self, input_features, attention_mask):
        return self.forward(input_features, attention_mask=attention_mask)


class BiLSTM_Linear_Pretrained(nn.Module):
    def __init__(self, input_size):
        super(BiLSTM_Linear_Pretrained, self).__init__()
        self.input_size = input_size

        self.lstm_part = BiLSTM_part(self.input_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # self.output_linear1 = torch.nn.Linear(1024, 512)
        # self.output_linear2 = torch.nn.Linear(512, 1)
        self.output_linear = torch.nn.Linear(128, 1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, input_features, attention_mask):
        lstm_output = self.lstm_part(input_features)

        embedding = []
        # 对lstm的输出进行降维到两维
        for seq_num in range(len(attention_mask)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = lstm_output[seq_num][0: seq_len]
            embedding.append(seq_emd)
        embedding = torch.cat(embedding, 0)

        output = self.sigmoid(self.output_linear(embedding))
        return output

    def predict(self, input_features, attention_mask):
        return self.forward(
            input_features=input_features,
            attention_mask=attention_mask
        )


class BiLSTM_Linear_Classification_Pretrained(nn.Module):
    def __init__(self, input_size):
        super(BiLSTM_Linear_Classification_Pretrained, self).__init__()
        self.input_size = input_size

        self.lstm_part = BiLSTM_part(self.input_size)
        self.relu = torch.nn.ReLU()

        self.output_linear1 = torch.nn.Linear(1024, 512)
        self.output_linear2 = torch.nn.Linear(512, 1)

        self.attention_layer1 = SimplifiedScaledDotProductAttention(d_model=1024, h=8)
        self.attention_layer2 = SimplifiedScaledDotProductAttention(d_model=1024, h=8)
        self.bn = torch.nn.BatchNorm1d(1024)

        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, input_features, attention_mask):
        lstm_output = self.lstm_part(input_features)
        att_output1 = self.relu(self.attention_layer1(lstm_output, lstm_output, lstm_output))
        att_output2 = self.relu(self.attention_layer2(att_output1, att_output1, att_output1))

        embedding = []
        # 对lstm的输出进行降维到两维
        for seq_num in range(len(attention_mask)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = att_output2[seq_num][0: seq_len]
            embedding.append(seq_emd)
        embedding = torch.cat(embedding, 0)

        embedding = self.bn(embedding)

        linear1 = self.relu(self.output_linear1(embedding))
        output = self.sigmoid(self.output_linear2(linear1))
        return output

    def predict(self, input_features, attention_mask):
        pre_pro = self.forward(
            input_features=input_features,
            attention_mask=attention_mask
        )
        # pre_pro = self.softmax(pre_pro)
        # pre_label = torch.argmax(pre_pro, dim=1)
        pre_label = torch.where(pre_pro > 0.2, 0, 1)
        return pre_label


class Basic_ProtBert(nn.Module):
    def __init__(self, model_path="../model/protBert-BFD/"):
        super(Basic_ProtBert, self).__init__()
        self.tokenizer, self.bert_model = get_model(model_path)

    def forward(self, input_embeds, attention_mask):
        embedding = self.bert_model(inputs_embeds=input_embeds, attention_mask=attention_mask)[0]
        return embedding


class BERT_BiLSTM(nn.Module):
    def __init__(self, input_size, bert_model_path, max_len, device):
        super(BERT_BiLSTM, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.BiLSTM = BiLSTM_part(input_size)

        self.input_linear = torch.nn.Linear(input_size, 1024)

        self.BiLSTM.load_state_dict(
            torch.load(
                "/public/home/yyang/cmq/access/methods/BERT_LSTM/02-26_model/BiLSTM_part_02-26_20_0.0002_Adam_54_pretrained BiLSTM.pth",
                map_location=device))

        self.bert_model = Basic_ProtBert(bert_model_path)
        self.attention_layer = DotProductAttention(.1)

        self.relu = torch.nn.ReLU()
        self.linear_bert = torch.nn.Linear(1024, 128)
        # self.output_linear2 = torch.nn.Linear(512, 1)
        self.output_linear = torch.nn.Linear(1152, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_features, attention_mask):
        input_embeds = self.relu(self.input_linear(input_features))

        lstm_output = self.BiLSTM(input_features)  # (batch_size, max_len, 1024)
        bert_output = self.bert_model(input_embeds, attention_mask)  # (batch_size, max_len, 1024)

        ##  点积融合
        # valid_len = []
        # for seq_num in range(len(attention_mask)):
        #     seq_len = (attention_mask[seq_num] == 1).sum()
        #     valid_len.append(seq_len)
        # valid_len = torch.tensor(valid_len)
        #
        # bert_output = self.relu(self.linear_bert(bert_output))
        #
        # attention_output = self.attention_layer(bert_output, lstm_output, lstm_output, valid_len)
        # ## 加性融合
        # bert_output = self.relu(self.linear_bert(bert_output))
        # attention_output = lstm_output + bert_output

        # concat融合
        attention_output = torch.cat((lstm_output, bert_output), dim=-1)

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


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

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

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

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
    # tm = Transformer_BiLSTM(50)
    # att = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    # a = torch.rand(2, 4, 50)
    # a = tm(a, att)
    # query = torch.rand(8, 512, 1024)
    # key = torch.rand(8, 512, 1024)
    # value = torch.rand(8, 512, 1024)
    # score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
    # att = masked_softmax(score, None)
    # c = torch.bmm(att, value)
    # print(1)
    # input1 = torch.randn(8, 512, 1024)
    # input2 = torch.randn(8, 512, 1024)
    # aft_full = AFT_FULL(d_model=1024, n=512)
    # output = aft_full(input1, input2)
    # print(output.shape)
    # blp = BiLSTM_part(50)
    # blp.load_state_dict(
    #     torch.load("05-19_model/BiLSTM_part_05-19_20_0.0002_Adam_50.pth", map_location=torch.device('cpu')))
    # print(1)
