# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/5/20
import torch
import datetime
import argparse

from bert_lstm_model import  BERT_BiLSTM_model
from torch.utils.data import DataLoader

import sys

sys.path.append("..")
from data_utils import AccessData, collate_fn, collate_fn2, collate_fn_for_bio

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
date_time = datetime.datetime.now().strftime("%m-%d")

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def load_data(path, batch_size, features_num, dataset, shuffle):
    access_data = AccessData(path, features_num=features_num,
                             dataset=dataset)
    access_dataloader = DataLoader(access_data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                   collate_fn=collate_fn)

    return access_dataloader

if __name__ == "__main__":
    print("运行日期为:{}".format(date_time))
    parser = argparse.ArgumentParser(description="显示程序搜索的超参数")
    parser.add_argument("--optimizer", type=str, help="优化器类型", default="Adam")
    parser.add_argument("--learning_rate", type=float, help="学习率", default=2e-4)
    parser.add_argument("--epoch", type=int, help="训练轮数", default=100)
    parser.add_argument("--features_num", type=int, help="使用特征的数量", default=50)
    parser.add_argument("--device", type=int, help="gpu序号", default=0)
    parser.add_argument("--description", type=str, help="说明", default="")

    args = parser.parse_args()

    print(args.description)

    batch_size = 16
    use_optuna = True
    features_num = args.features_num
    threshold = 20
    # optimizer = ["RMSprop", "Adam"]
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    epoch_num = args.epoch

    # train_path = "../test_data_50.csv"
    train_path = "../train_data_50.csv"
    # validation_path = "../test_data_50.csv"
    validation_path = "../validation_data_50.csv"
    test_path = "../test_data_50.csv"

    test_dataloader = load_data(path=test_path, batch_size=batch_size, dataset="test",
                                features_num=features_num, shuffle=False)
    train_dataloader = load_data(path=train_path, batch_size=batch_size, dataset="train",
                                 features_num=features_num, shuffle=True)
    validation_dataloader = load_data(path=validation_path, batch_size=batch_size,
                                      dataset="validation", features_num=features_num, shuffle=False)
    model = BERT_BiLSTM_model(
             input_size=features_num,
             learning_rate=learning_rate,
             optimizer=optimizer,
             epoch_nums=epoch_num,
             features_num=features_num,
             device=args.device,
             description=args.description,
             is_save=False,
             is_csv=False,
             test_path="../test_data_50.csv"
        )

        model.train_and_validation(train_dataloader, validation_dataloader, test_dataloader)

    
