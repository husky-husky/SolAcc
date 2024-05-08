# coding: utf-8
# @Author    :陈梦淇
# @time      :2022/8/21
import torch
import logging
from logging import handlers


class LRScheduler:
    def __init__(
            self, optimizer, patience=10, min_lr=1e-6, factor=0.5
    ):
        """
        初始化，学习率更新策略:new_lr = old_lr * factor
        :param optimizer: 优化器
        :param patience: 容忍验证集损失不发生变化的最大轮数
        :param min_lr: 最小的学习率
        :param factor: 学习率更新因子
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        """

        :param patience:
        :param min_delta: 上一轮损失和此轮损失的阈值界限
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter = self.counter + 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式

    stream_handler = logging.StreamHandler()  # 设置往屏幕上输出
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    th = logging.FileHandler(log_name)
    th.setFormatter(formatter)
    logger.addHandler(th)

    return logger



