# -*- coding: utf-8 -*-
__author__ = 'lijingjie'

import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/')
sys.path.insert(0, './')
sys.path.insert(0, 'conf/')
import numpy as np
import pandas as pd
from utils import Dataset
from keras.callbacks import Callback
categoricals = Dataset.get_part_features('categorical')
print(categoricals)

class CategoricalAlphaEncoded(object):

    requirements = ['categorical']

    def __init__(self, combinations=[]):
        self.combinations = [list(map(categoricals.index, comb)) for comb in combinations]

    def fit_transform(self, ds):
        return self.transform(ds)

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in range(len(categoricals)):
            test_res[:, col] = self.transform_column(test_cat[:, col])

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            # print ('ljj is ',map(''.join, test_cat[:, comb])) # 就是对类别做了string 拼接然后做了个排序吧
            test_res[:, col] = self.transform_column(list(map(''.join, test_cat[:, comb])))
        return test_res

    def transform_column(self, arr):
        def encode(charcode):
            r = 0
            ln = len(charcode)
            for i in range(ln):
                r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)  ####??????
            return r

        return np.array(list(map(encode, arr)))

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]

class CategoricalMeanEncoded(object):

    requirements = ['categorical']

    def __init__(self, C=100, loo=False, noisy=True, noise_std=None, random_state=11, combinations=[]):
        self.random_state = np.random.RandomState(random_state)
        self.C = C
        self.loo = loo
        self.noisy = noisy
        self.noise_std = noise_std
        self.combinations = [list(map(categoricals.index, comb)) for comb in combinations]

    def fit_transform(self, ds):
        train_cat = ds['categorical']
        train_target = pd.Series(np.log(ds['loss'] + 100))
        train_res = np.zeros((train_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std = train_target.std() if self.noise_std is None else self.noise_std

        self.target_sums = {}
        self.target_cnts = {}

        for col in range(len(categoricals)):
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(train_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(list(map(''.join, train_cat[:, comb]))))

        return train_res

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in range(len(categoricals)):
            test_res[:, col] = self.transform_column(col, pd.Series(test_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            test_res[:, col] = self.transform_column(col, pd.Series(list(map(''.join, test_cat[:, comb]))))

        return test_res

    def fit_transform_column(self, col, train_target, train_series):
        self.target_sums[col] = train_target.groupby(train_series).sum()
        self.target_cnts[col] = train_target.groupby(train_series).count()

        if self.noisy:
            train_res_reg = self.random_state.normal(
                loc=self.global_target_mean * self.C,
                scale=self.global_target_std * np.sqrt(self.C),
                size=len(train_series)
            )
        else:
            train_res_reg = self.global_target_mean * self.C

        train_res_num = train_series.map(self.target_sums[col]) + train_res_reg
        train_res_den = train_series.map(self.target_cnts[col]) + self.C

        if self.loo:  # Leave-one-out mode, exclude current observation
            train_res_num -= train_target
            train_res_den -= 1

        return np.exp(train_res_num / train_res_den).values

    def transform_column(self, col, test_series):
        test_res_num = test_series.map(self.target_sums[col]).fillna(0.0) + self.global_target_mean * self.C
        test_res_den = test_series.map(self.target_cnts[col]).fillna(0.0) + self.C

        return np.exp(test_res_num / test_res_den).values

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]

class ExponentialMovingAverage(Callback):
    """create a copy of trainable weights which gets updated at every
       batch using exponential weight decay. The moving average weights along
       with the other states of original model(except original model trainable
       weights) will be saved at every epoch if save_mv_ave_model is True.
       If both save_mv_ave_model and save_best_only are True, the latest
       best moving average model according to the quantity monitored
       will not be overwritten. Of course, save_best_only can be True
       only if there is a validation set.
       This is equivalent to save_best_only mode of ModelCheckpoint
       callback with similar code. custom_objects is a dictionary
       holding name and Class implementation for custom layers.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of the training, the moving
       averaged weights are transferred to the original model.
       """
    def __init__(self, decay=0.999, filepath='temp_weight.hdf5',
                 save_mv_ave_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto',
                 save_weights_only=False, custom_objects={}):
        self.decay = decay
        self.filepath = filepath
        self.verbose = verbose
        self.save_mv_ave_model = save_mv_ave_model
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.custom_objects = custom_objects  # dictionary of custom layers
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        super(ExponentialMovingAverage, self).__init__()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        self.sym_trainable_weights = collect_trainable_weights(self.model)
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}
        if self.verbose:
            print('Created a copy of model weights to initialize moving'
                  ' averaged weights.')

    def on_batch_end(self, batch, logs={}):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we can optionally save the moving averaged model,
        but the weights will NOT be transferred to the original model. This
        happens only at the end of training. We also need to transfer state of
        original model to model2 as model2 only gets updated trainable weight
        at end of each batch and non-trainable weights are not transferred
        (for example mean and var for batch normalization layers)."""
        if self.save_mv_ave_model:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(('saving moving average model to %s'
                                  % (filepath)))
                        self.best = current
                        model2 = self._make_mv_model(filepath)
                        if self.save_weights_only:
                            model2.save_weights(filepath, overwrite=True)
                        else:
                            model2.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print(('Epoch %05d: saving moving average model to %s' % (epoch, filepath)))
                model2 = self._make_mv_model(filepath)
                if self.save_weights_only:
                    model2.save_weights(filepath, overwrite=True)
                else:
                    model2.save(filepath, overwrite=True)

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self, filepath):
        """ Create a model with moving averaged weights. Other variables are
        the same as original mode. We first save original model to save its
        state. Then copy moving averaged weights over."""
        self.model.save(filepath, overwrite=True)
        model2 = load_model(filepath, custom_objects=self.custom_objects)

        for w2, w in zip(collect_trainable_weights(model2), collect_trainable_weights(self.model)):
            K.set_value(w2, self.mv_trainable_weights_vals[w.name])

        return model2