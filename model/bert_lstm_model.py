# coding: utf-8
# @Author    :陈梦淇
# @time      :2023/5/20
import os.path

import torch
import pandas as pd
import numpy as np

import time
import datetime

from BERT_LSTM import Transformer_BiLSTM
from torch.nn.parallel import DataParallel

import sys

sys.path.append("..")
from utils import LRScheduler, EarlyStopping, get_logger
from metrics import metrics_classification, metrics_regression, metrics_mae

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
from torch import optim

date_time = datetime.datetime.now().strftime("%m-%d")


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


class BERT_BiLSTM_model(torch.nn.Module):
    def __init__(self, input_size, learning_rate, optimizer, epoch_nums, dropout=0, early_stop=True,
                 lr_scheduler=True, threshold=20, use_features=True, features_num=0,
                 device=0, description="", is_save=False, test_path=None, is_csv=False):
        super(BERT_BiLSTM_model, self).__init__()

        self.threshold = threshold
        self.input_size = input_size
        self.epoch_nums = epoch_nums
        self.dropout = dropout
        self.use_features = use_features
        self.features_num = features_num
        self.description = description
        self.is_save = is_save
        self.test_path = test_path
        self.is_csv = is_csv

        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")

        if self.use_features:
            # self.model = Transformer_BiLSTM(input_size=self.features_num)
            self.model = Transformer_BiLSTM(input_size=self.features_num)

        self.learning_rate = learning_rate  # 学习率
        self.optimizer_type = optimizer
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=self.learning_rate)  # 优化器

        # 定义损失函数
        self.loss_fn = torch.nn.L1Loss()

        if lr_scheduler:
            self.lr_scheduler = LRScheduler(self.optimizer)  # 学习率优化器
        else:
            self.lr_scheduler = None
        if early_stop:
            self.early_stopping = EarlyStopping()  # 早停策略
        else:
            self.early_stopping = None

        if not os.path.exists("{}_model".format(date_time)):
            os.mkdir("{}_model".format(date_time))

        log_name = "{}_model/BiLSTM_WITH_FEATURES{}_{}_{}_{}.log".format(date_time,
                                                                         date_time, learning_rate,
                                                                         optimizer, self.description)
        self.logger = get_logger(log_name)

        self.logger.info("损失函数是L1Loss，学习率:{}，优化器{}".format(self.learning_rate, optimizer))

    def train_part(self, train_dataloader, epoch):
        total_loss_train = 0

        t0 = time.time()
        self.model.train()
        for step, (inputs, attention_masks, labels, aas) in enumerate(train_dataloader):
            if step % 100 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                self.logger.info(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            inputs, attention_masks, labels = inputs.to(self.device), attention_masks.to(self.device), labels.to(
                self.device)
            y_true = []
            for seq_num in range(len(labels)):
                seq_len = (attention_masks[seq_num] == 1).sum()
                label = labels[seq_num][0:seq_len]
                y_true.append(label)

            y_true = torch.cat(y_true, 0)

            self.optimizer.zero_grad()
            y_pred = self.model.forward(input_features=inputs, attention_mask=attention_masks)
            y_pred = y_pred.view(-1, )
            # label = label.view(-1, )
            loss = self.loss_fn(y_pred, y_true)
            loss.backward()

            self.optimizer.step()

            total_loss_train = total_loss_train + loss.cpu().detach().numpy()

        training_time = format_time(time.time() - t0)
        self.logger.info("")
        self.logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
        self.logger.info("  Training epcoh took: {:}".format(training_time))

        return total_loss_train / len(train_dataloader)

    def validation_part(self, validation_dataloader, epoch):
        total_loss_val = 0
        pre_labels = []
        real_labels = []
        self.model.eval()
        total_num = 0
        with torch.no_grad():
            for inputs, attention_masks, labels, aas in validation_dataloader:
                inputs, attention_masks, labels = inputs.to(self.device), attention_masks.to(
                    self.device), labels.to(self.device)
                y_true = []
                for seq_num in range(len(labels)):
                    seq_len = (attention_masks[seq_num] == 1).sum()

                    label = labels[seq_num][0:seq_len]
                    y_true.append(label)

                y_true = torch.cat(y_true, 0)
                output = self.model.forward(inputs, attention_masks)

                output = output.view(-1, )

                val_loss = self.loss_fn(output, y_true)
                total_loss_val = total_loss_val + val_loss

                pre_labels.extend(output.detach().cpu().numpy().tolist())
                real_labels.extend(y_true.detach().cpu().numpy().tolist())

        # mae = metrics_mae(np.array(pre_labels), np.array(real_labels))
        pre_labels, real_labels = np.array(pre_labels), np.array(real_labels)
        mae, pcc, mse, rmse, msle, r2, classification_performance = metrics_regression(pre_labels, real_labels)
        self.logger.info("============Performance on validation==============")
        self.logger.info(
            "MAE:{:.4f}, PCC:{:.4f}, MSE:{:.4f}, rMSE:{:.4f}, MSLE:{:.4f}, r2:{:.4f}".format(float(mae), float(pcc),
                                                                                             float(mse)
                                                                                             , float(rmse), float(msle),
                                                                                             float(r2)))

        threshold = [5, 10, 20, 25, 30, 40, 50]
        for index, row in classification_performance.iterrows():
            self.logger.info("Threshold:{}, precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, "
                             "mcc:{:.4f}, "
                             "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                             .format(threshold[index], row["precision"], row["recall"], row["accuracy"], row["F1"],
                                     row["auc_score"], row["mcc"], row["ppv"], row["npv"], row["tpr"], row["tnr"]))

        self.logger.info("  Average validation loss: {0:.6f}".format(total_loss_val / len(validation_dataloader)))
        self.logger.info("  Epoch:{},ACC on validation:{:.4f}".format(epoch + 1, mae))
        return mae, total_loss_val / len(validation_dataloader)

    def train_and_validation(self, train_dataloader, validation_dataloader, test_dataloader):
        """
        :param test_dataloader:
        :param train_dataloader:
        :param validation_dataloader:
        :return:
        """
        train_acc, train_loss = 0, 0
        best_val_mae = 10
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        torch.cuda.empty_cache()

        # self.model.initialize_weights()

        t0 = time.time()
        self.logger.info("==================description:{}====================".format(self.description))
        self.logger.info("========learning_rate:{:}=========".format(self.learning_rate))
        for epoch in range(self.epoch_nums):
            self.logger.info("")
            self.logger.info('======== Epoch {:} / {:} ========'.format(epoch + 1, self.epoch_nums))
            self.logger.info('Training...')

            train_loss = self.train_part(train_dataloader, epoch)
            val_mae, val_loss = self.validation_part(validation_dataloader, epoch)
            self.test(test_dataloader, epoch)
            if self.is_save:
                self.save_model(epoch + 1)

            # # 如果模型在验证集上表现效果不错，则测试其在盲测集上的性能
            # if val_mae < best_val_mae:
            #     best_val_mae = val_mae
            #     self.test(test_dataloader)

            if self.lr_scheduler:
                self.lr_scheduler(val_loss)
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    break

            # 计算GPU利用率
            total_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory, _ = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            memory_utilization = used_memory / total_memory * 100

            self.logger.info(
                f"Used / Total Memory: {used_memory}/{total_memory}. Utilization: {memory_utilization:.2f}%")

        self.logger.info("Training complete!")
        self.logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - t0)))

        self.logger.info("==========train accuracy: {}==========".format(train_acc))
        self.logger.info("==========validation mae: {}==========".format(best_val_mae))

        self.save_model("final")
        return best_val_mae

    def save_model(self, epoch):
        if not os.path.exists("{}_model".format(date_time)):
            os.mkdir("{}_model".format(date_time))

        # 保存模型
        torch.save(self.model.state_dict(),
                   "{}_model/BiLSTM{}_{}_{}_{}_{}.pth".format(date_time, date_time, self.description,
                                                              self.learning_rate, self.optimizer_type,
                                                              epoch))

    def test(self, dataloader, epoch):
        pre_labels = []
        real_labels = []
        aa_list = []
        self.model.eval()
        with torch.no_grad():
            for inputs, attention_masks, labels, aas in dataloader:
                inputs, attention_masks, labels = inputs.to(self.device), attention_masks.to(
                    self.device), labels.to(self.device)
                y_true = []
                for seq_num in range(len(labels)):
                    seq_len = (attention_masks[seq_num] == 1).sum()
                    label = labels[seq_num][0:seq_len]
                    y_true.append(label)

                y_true = torch.cat(y_true, 0)
                pre = self.model.predict(inputs, attention_masks)
                pre = pre.view(-1, )

                pre_labels.extend(pre.detach().cpu().numpy().tolist())
                real_labels.extend(y_true.detach().cpu().numpy().tolist())
                aa_list.extend(aas)

        pre_labels, real_labels = np.array(pre_labels), np.array(real_labels)
        mae, pcc, mse, rmse, msle, r2, classification_performance = metrics_regression(pre_labels, real_labels)
        self.logger.info("============Performance on test==============")
        self.logger.info(
            "MAE:{:.4f}, PCC:{:.4f}, MSE:{:.4f}, rMSE:{:.4f}, MSLE:{:.4f}, r2:{:.4f}".format(float(mae), float(pcc),
                                                                                             float(mse), float(rmse),
                                                                                             float(msle), float(r2)))

        threshold = [5, 10, 20, 25, 30, 40, 50]
        for index, row in classification_performance.iterrows():
            self.logger.info("Threshold:{}, precision:{:.4f}, recall:{:.4f}, accuracy:{:.4f}, F1:{:.4f}, auc:{:.4f}, "
                             "mcc:{:.4f}, "
                             "ppv:{:.4f}, npv:{:.4f}, tpr:{:.4f}, tnr:{:.4f} "
                             .format(threshold[index], row["precision"], row["recall"], row["accuracy"], row["F1"],
                                     row["auc_score"], row["mcc"], row["ppv"], row["npv"], row["tpr"], row["tnr"]))

        df = pd.DataFrame(list(zip(aa_list, pre_labels, real_labels)), columns=["aa", "pre_label", "real_label"])
        if self.is_csv:
            df.to_csv(f"{date_time}_model/performance_test_{epoch}.csv")

   
    def predict(self, x, attention_mask):
        return self.model.predict(x, attention_mask)
