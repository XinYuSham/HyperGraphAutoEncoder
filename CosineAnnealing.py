import tensorflow as tf
from tensorflow import keras
import numpy as np


class CosineAnnealing(keras.callbacks.Callback):
    def __init__(self, eta_max=1, eta_min=0, total_iteration=0, iteration=0, verbose=0, **kwargs):
        super(CosineAnnealing, self).__init__()
        global lr_list
        lr_list = []
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.total_iteration = total_iteration
        self.iteration = iteration

    def on_train_begin(self, logs=None):
        self.lr = tf.keras.backend.get_value(self.model.optimizer.lr)

    def on_train_end(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_batch_end(self, epoch, logs=None):
        self.iteration += 1
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

        eta_t = self.eta_min + (self.eta_max - self.eta_min) * 0.5 * (
                    1 + np.cos(np.pi * self.iteration / self.total_iteration))
        new_lr = self.lr * eta_t
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose > 0:
            print('/nEpoch %05d: CosineAnnealing '
                  'learning rate to %s.' % (epoch + 1, new_lr))
        lr_list.append(logs['lr'])
