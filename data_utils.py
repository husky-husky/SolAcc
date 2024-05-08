# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/8/2
from abc import ABC
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer


class AccessData(Dataset):
    def __init__(self, file_path, features_num, dataset, min_len=32, max_len=512):
        super(AccessData, self).__init__()
        self.features_num = features_num  # 特征数量
        pis_data = pd.read_csv(file_path)

        self.min_len, self.max_len = min_len, max_len
        self.dataset = dataset
        self.file_path = file_path

        features_importance = pd.read_csv("../../features_select/features_importance_all_train_data.csv")
        columns = features_importance.columns.tolist()
        importance = features_importance.loc[0].tolist()
        features2importance = dict(zip(columns, importance))
        features2importance_sorted = sorted(features2importance.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [features2importance_sorted[i][0] for i in range(self.features_num)]

        self.bio_features, self.labels, self.seq_len, self.aa_list = self.get_features()

    def __getitem__(self, item):
        output = {
            "bio_features": torch.tensor(self.bio_features[item]),
            "labels": torch.tensor(self.labels[item]),
            "seq_len": self.seq_len[item],
            "fea_num": len(self.selected_features),
            "aa_list": self.aa_list[item]
        }
        return output

    def __len__(self):
        return len(self.bio_features)

    def get_features(self):
        bio_features, labels, seq_len, aa_list = [], [], [], []

        all_data = pd.read_csv(self.file_path)
        all_data["id_chain"] = all_data["id"] + all_data["chain"]
        all_data_groups = all_data.groupby("id_chain")

        for name, group in tqdm(all_data_groups, desc=f"loading {self.dataset}"):
            if len(group) > 512:
                group = group.iloc[:512]
            aa_list.append(group["aa"].values.tolist())
            bio_fea = group[self.selected_features].values.tolist()
            label = group["new_rsa"].values.tolist()
            bio_features.append(bio_fea)
            labels.append(label)
            seq_len.append(len(group))

        return bio_features, labels, seq_len, aa_list


class AccessDataForRNN(Dataset):
    def __init__(self, file_path, threshold, min_len, max_len, use_features=False, features_num=0):
        super(AccessDataForRNN, self).__init__()

        self.file_path = file_path
        self.threshold = threshold
        self.min_len = min_len
        self.max_len = max_len
        self.features_num = features_num
        self.use_features = use_features

        features_importance = pd.read_csv("../../features_select/features_importance_all_train_data.csv")
        columns = features_importance.columns.tolist()
        importance = features_importance.loc[0].tolist()
        features2importance = dict(zip(columns, importance))
        features2importance_sorted = sorted(features2importance.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [features2importance_sorted[i][0] for i in range(self.features_num)]

        self.data, self.label = self.get_data()
        self.nums = self.get_data_nums()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

    def get_data(self):
        data = pd.read_csv(self.file_path)
        cols = self.selected_features

        seq_len = np.array(data["id"].value_counts())
        seq_id_chain = np.array(data["id"].value_counts().index)

        features, labels = [], []
        for i in range(0, len(seq_len)):
            a = data[data.id == seq_id_chain[i]]
            if self.min_len <= seq_len[i] <= self.max_len:
                features.append(np.array(a[cols], dtype=np.float32))
                labels.append(np.array(a["new_rsa"]))
            if seq_len[i] >= self.max_len:
                features.append(np.array(a[cols], dtype=np.float32)[:self.max_len])
                labels.append(np.array(a["new_rsa"])[:self.max_len])
        return features, labels

        # id_chains = []
        #
        # for i in range(0, len(seq_len)):
        #     if self.min_len <= seq_len[i] <= self.max_len:
        #         id_chains.append(seq_id_chain[i])
        #
        # # label = np.array(data["label_{}".format(self.threshold)])
        #
        # features, labels = [], []
        # for id_chain in id_chains:
        #     a = data[data.id_chain == id_chain]
        #     # cols = list(a.columns.values)
        #     # cols = cols[3: 646]
        #
        #     a_data = np.array(a[cols], dtype=np.float32)
        #     # a_label = np.array(a["label_{}".format(self.threshold)])
        #     a_label = np.array(a["rsa"])
        #
        #     features.append(a_data)
        #     labels.append(a_label)
        #
        # return features, labels

    def get_data_pssm(self):
        pssm_aa_list = list("ARNDCQEGHILKMFPSTWYV")
        pssm_cols = ["{}_y".format(aa) for aa in pssm_aa_list]
        data = pd.read_csv(self.file_path)
        seq_len = np.array(data["id_chain"].value_counts())
        seq_id_chain = np.array(data["id_chain"].value_counts().index)

        id_chain = []
        for i in range(0, len(seq_len)):
            if self.min_len <= seq_len[i] <= self.max_len:
                id_chain.append(seq_id_chain[i])

        features, labels = [], []
        for id_chain in id_chain:
            a = data[data.id_chain == id_chain]
            cols = ["len_of_fas", "pos_in_fas"]
            cols.extend(pssm_cols)

            a_data = np.array(a[cols], dtype=np.float32)
            # a_label = np.array(a["label_{}".format(self.threshold)])
            a_label = np.array(a["rsa"])

            features.append(a_data)
            labels.append(a_label)

        return features, labels

    def get_data_nums(self):
        length = 0
        for i in self.label:
            length = length + len(i)
        return length


class AccessDataForBERT(Dataset):
    def __init__(self, file_path, min_len, max_len, dataset, features_num=0, tokenizer_path="model/protBert-BFD/",
                 use_features=False, use_features_and_seq=0):
        """
        :param use_features_and_seq: 0-仅使用生物特征 1-仅使用序列特征 2-同时使用到序列特征和生物特征
        """
        super(AccessDataForBERT, self).__init__()
        self.file_path = file_path
        self.min_len = min_len
        self.max_len = max_len
        self.tokenizer_path = tokenizer_path
        self.use_features = use_features  # 是否使用特征
        self.features_num = features_num  # 特征数量
        self.use_features_and_seq = use_features_and_seq
        self.dataset = dataset  # 数据集类型

        self.file_data = pd.read_csv(self.file_path)  # 读取原始数据文件
        self.filter_id = self.filter_seq()  # 得到符合长度条件的id

        # 读取生物特征的重要性，并进行排序
        features_importance = pd.read_csv("../../features_select/features_importance_all_train_data.csv")
        columns = features_importance.columns.tolist()
        importance = features_importance.loc[0].tolist()
        features2importance = dict(zip(columns, importance))
        features2importance_sorted = sorted(features2importance.items(), key=lambda x: x[1], reverse=True)

        if self.use_features_and_seq == 0:  # 仅使用生物特征，手动生成attention_mask
            self.selected_features = [features2importance_sorted[i][0] for i in range(self.features_num)]
            self.bio_features, self.labels, self.seq_len = self.get_bio_features()
        elif self.use_features_and_seq == 1:  # 仅使用序列特征
            self.seq_data, self.labels = self.generate_seq()
            self.input_ids, self.attention_masks = self.get_ids_mask()
        elif self.use_features_and_seq == 2:  # 需要同时使用到序列特征和生物特征
            # 获得生物特征，同时获得标签和attention_mask
            self.selected_features = [features2importance_sorted[i][0] for i in range(self.features_num)]
            self.input_embeds, self.attention_masks, self.labels = self.get_input_mask_labels()

            # 获得tokenizer的input_ids和attention_mask
            self.seq_data, self.labels = self.generate_seq()
            self.input_ids, self.attention_masks = self.get_ids_mask()

    def __getitem__(self, item):
        if self.use_features_and_seq == 0:  # 注意返回的是手动生成的attention_mask
            return {
                "bio_features": torch.tensor(self.bio_features[item]),
                "labels": torch.tensor(self.labels[item]),
                "seq_len": self.seq_len[item],
                "fea_num": len(self.selected_features)
            }
        elif self.use_features_and_seq == 1:  # 注意返回的是tokenizer的attention_mask
            return torch.tensor(self.input_embeds[item]), torch.tensor(self.attention_masks[item]), torch.tensor(
                self.labels[item])
        elif self.use_features_and_seq == 2:  # 注意返回的是tokenizer的attention_mask
            return torch.as_tensor(self.input_embeds[item]), torch.as_tensor(
                self.attention_masks[item]), torch.as_tensor(
                self.labels[item]), torch.as_tensor(self.input_ids[item])

    def __len__(self):
        return len(self.labels)

    def filter_seq(self):
        """
        filter based on min and max length
        :return:
        """
        seq_len = np.array(self.file_data["id"].value_counts())
        seq_id_chain = np.array(self.file_data["id"].value_counts().index)

        filter_id_chain = {}  # dict: key=id, value=is_truncated
        for i in range(0, len(seq_len)):
            if self.min_len <= seq_len[i] <= self.max_len:
                filter_id_chain[seq_id_chain[i]] = False
            elif seq_len[i] > self.max_len:
                filter_id_chain[seq_id_chain[i]] = True

        return filter_id_chain

    def generate_seq(self):
        seq_data = []
        labels = []
        for seq_id, is_truncated in self.filter_id.items():
            a = self.file_data[self.file_data.id == seq_id]
            aa, rsa = a["aa"].values.tolist(), a["new_rsa"].values.tolist()
            if not is_truncated:  # 符合长度条件不需要截断但需要补齐
                seq = "".join(aa)
                seq_data.append(" ".join(seq))
                rsa.extend([-1] * (self.max_len - len(rsa)))
                labels.append(rsa)
            else:  # 超过了最大长度，需要截断
                aa, rsa = aa[:self.max_len], rsa[:self.max_len]
                seq = "".join(aa)
                seq_data.append(" ".join(seq))
                labels.append(rsa)

        return seq_data, labels

    def get_ids_mask(self):
        input_ids = []
        attention_masks = []
        for seq in self.seq_data:
            seq = re.sub(r"[UZOB]", "X", seq)
            ids = self.tokenizer.encode_plus(seq, add_special_tokens=False, return_attention_mask=True,
                                             padding="max_length", max_length=self.max_len)
            input_ids.append(torch.tensor(ids["input_ids"]))
            attention_masks.append(torch.tensor(ids["attention_mask"]))

        return input_ids, attention_masks

    def get_bio_features(self):
        bio_features, labels, seq_len = [], [], []
        data_groups = self.file_data.groupby("id")
        data_dict = {}
        for name, group in tqdm(data_groups, desc=f"loading {self.dataset}"):
            if len(group) > self.max_len:
                group = group.iloc[: self.max_len]
            data_dict[name] = group

            bio_features.append(group[self.selected_features].values.tolist())
            labels.append(group["new_rsa"].values.tolist())
            seq_len.append(len(group))
        return bio_features, labels, seq_len

    def get_input_mask_labels(self):
        inputs_embeds = []
        attention_masks = []
        labels = []
        for seq_id, is_truncated in self.filter_id.items():
            if not is_truncated:  # 长度符合条件，不需要截断，直接添加
                a = self.file_data[self.file_data.id == seq_id]
                # 读取特征，并用0补齐，对于一条序列：（self.max_len, self.features_num）
                features = a[self.selected_features].values.tolist()
                original_len = len(features)
                features.extend([[0] * self.features_num] * (self.max_len - original_len))
                inputs_embeds.append(features)

                # 生成mask
                masks = [1] * original_len
                masks.extend([0] * (self.max_len - original_len))
                attention_masks.append(masks)

                # 读取label，不足的用0补齐
                rsa = a["new_rsa"].values.tolist()
                rsa.extend([0] * (self.max_len - original_len))
                labels.append(rsa)
            else:  # 长度超过设定的最大长度，对其进行截断
                a = self.file_data[self.file_data.id == seq_id]
                # 读取生物特征
                features = a[self.selected_features].values.tolist()
                features = np.array(features, dtype=np.float32)[:self.max_len]
                inputs_embeds.append(features)

                # 生成mask
                masks = [1] * len(features)
                attention_masks.append(masks)

                # 读取label
                rsa = a["new_rsa"].values.tolist()
                rsa = rsa[: self.max_len]
                labels.append(rsa)

        return inputs_embeds, attention_masks, labels

    def get_input_mask_labels_seq(self):
        inputs_embeds = []
        attention_masks = []
        labels = []
        seq_list = []

        data = pd.read_csv(self.file_path)  # 所有特征

        seq_len = np.array(data["id"].value_counts())
        seq_id_chain = np.array(data["id"].value_counts().index)

        id_chains = []
        for i in range(0, len(seq_len)):
            if self.min_len <= seq_len[i] <= self.max_len:
                id_chains.append(seq_id_chain[i])

        for id_chain in id_chains:
            a = data[data.id == id_chain]
            # 读取特征，并用0补齐，对于一条序列：（self.max_len, self.features_num）
            features = a[self.selected_features].values.tolist()
            original_len = len(features)
            features.extend([[0] * self.features_num] * (self.max_len - original_len))
            inputs_embeds.append(features)

            # 生成mask
            masks = [1] * original_len
            masks.extend([0] * (self.max_len - original_len))
            attention_masks.append(masks)

            # 读取label，不足的用0补齐
            rsa = a["new_rsa"].values.tolist()
            rsa.extend([0] * (self.max_len - original_len))
            labels.append(rsa)

            seq = a["aa"].values.tolist()
            seq_str = ""
            for aa in seq:
                seq_str = seq_str + aa + " "
            seq_list.append(seq_str.strip())

        return inputs_embeds, attention_masks, labels, seq_list

    def get_data(self):
        data = pd.read_csv(self.file_path)
        seq_len = np.array(data["id"].value_counts())
        seq_id_chain = np.array(data["id"].value_counts().index)

        id_chains = []
        for i in range(0, len(seq_len)):
            if self.min_len <= seq_len[i] <= self.max_len:
                id_chains.append(seq_id_chain[i])

        seqs, labels = [], []
        for id_chain in id_chains:
            a = data[data.id_chain == id_chain]
            a_label = np.array(a["new_rsa"])
            a_data = np.array(a["aa"])

            seq = ""
            for i in range(0, len(a_data) - 1):
                seq = seq + a_data[i] + " "
            seq = seq + a_data[-1]

            seqs.append(seq)
            labels.append(a_label)

        return seqs, labels


def collate_fn(batch):
    bio_features = [data["bio_features"] for data in batch]
    labels = [data["labels"] for data in batch]
    max_len = max([data["seq_len"] for data in batch])
    fea_num = batch[0]["fea_num"]
    aa_list = [data["aa_list"] for data in batch]
    aas = []
    for aal in aa_list:
        aas.extend(aal)

    attention_mask = [torch.cat((torch.tensor([1] * len(item)), torch.tensor([0] * (max_len - len(item)))), -1)
                      for item in bio_features]
    attention_mask = torch.stack(attention_mask, dim=0)

    # padding bio features
    for i in range(len(bio_features)):
        padding_zero = torch.zeros([max_len - bio_features[i].shape[0], fea_num], dtype=torch.float)
        bio_features[i] = torch.cat((bio_features[i], padding_zero), 0)
    bio_features = torch.stack(bio_features)

    # padding labels
    labels = [torch.cat((item, torch.tensor([0] * (max_len - len(item)))), -1) for item in labels]
    labels = torch.stack(labels, dim=0)

    return bio_features, attention_mask, labels, aas


def collate_fn2(train_data):
    train_data.sort(key=lambda x: len(x[0]), reverse=True)
    data = [torch.tensor(i[0], dtype=torch.float32) for i in train_data]
    label = [torch.tensor(i[1], dtype=torch.float32) for i in train_data]
    data_length = [len(sq) for sq in data]

    data = pad_sequence(data, batch_first=True, padding_value=0.0)  # 用零补充，使长度对齐
    label = pad_sequence(label, batch_first=True, padding_value=0.0)  # 这行代码只是为了把列表变为tensor

    return data, label, data_length


def collate_fn_for_bio(batch):
    bio_features_num = batch[0]["fea_num"]
    bio_feature = [data["bio_features"] for data in batch]
    seq_len = [data["seq_len"] for data in batch]
    labels = [data["labels"] for data in batch]
    max_len = max(seq_len)
    # generate mask: attention_mask
    attention_mask = [torch.cat([torch.tensor([1] * item), torch.tensor([0] * (max_len - item))], -1)
                      for item in seq_len]
    attention_mask = torch.stack(attention_mask, dim=0)

    # padding labels
    labels = [torch.cat([item, torch.tensor([0] * (max_len - len(item)))], -1) for item in labels]
    labels = torch.stack(labels, dim=0)

    # padding bio features
    for i in range(len(bio_feature)):
        padding_zero = torch.zeros([max_len - bio_feature[i].shape[0], bio_features_num], dtype=torch.float)
        bio_feature[i] = torch.cat([bio_feature[i], padding_zero], 0)
    bio_feature = torch.stack(bio_feature)

    return bio_feature, attention_mask, labels


def one_hot(label):
    """
    将标签转换成onehot编码形式
    :param label: numpy
    :return:
    """
    one_hot_label = []
    for i in label:
        temp = [0, 0]
        if i != 0.5:
            temp[i] = 1
            one_hot_label.append(temp)
        else:
            one_hot_label.append([0.5, 0.5])
    return np.array(one_hot_label)


if __name__ == "__main__":
    pass
   
