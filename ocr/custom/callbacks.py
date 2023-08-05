import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import Callback


class HistoryCache:

    def __init__(self, his_len=10):
        self.history = [0] * his_len
        self.history_len = his_len
        self.cursor = 0
        self.len = 0

    def put(self, value):
        self.history[self.cursor] = value
        self.cursor += 1
        if self.cursor >= self.history_len:
            self.cursor = 0
        if self.len + 1 <= self.history_len:
            self.len += 1

    def mean(self):
        return np.array(self.history[0: self.len]).mean()


class LRScheduler(Callback):

    def __init__(self, schedule, watch, watch_his_len=10):
        super().__init__()
        self.schedule = schedule
        self.watch = watch
        self.history_cache = HistoryCache(watch_his_len)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        watch_value = logs.get(self.watch)
        if watch_value is None:
            raise ValueError("Watched value '" + self.watch + "' don't exist")

        self.history_cache.put(watch_value)

        if watch_value > self.history_cache.mean():
            lr = self.schedule(epoch, lr)
            print("Update learning rate: ", lr)
            K.set_value(self.model.optimizer.lr, lr)
