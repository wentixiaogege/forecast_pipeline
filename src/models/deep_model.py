# -*- coding: utf-8 -*-
__author__ = 'lijingjie'

import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/')
sys.path.insert(0, '../conf')
import os
import numpy as np
import scipy.sparse as sp

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K

class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None,**kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            print 'print val[0].shape',val[0].shape
            print 'print test[0].shape',test[0].shape
            return self.predict(val[0]), self.predict(test[0])

def batch_generator(X, y=None, batch_size=128, shuffle=False):
    index = np.arange(X.shape[0])

    while True:
        if shuffle:
            np.random.shuffle(index)

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_index = index[batch_start:batch_start + batch_size]
            batch_start += batch_size

            X_batch = X[batch_index, :]

            if sp.issparse(X_batch):
                X_batch = X_batch.toarray()

            if y is None:
                yield X_batch
            else:
                yield X_batch, y[batch_index]


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Keras(BaseAlgo):


    def __init__(self, arch, params, scale=True, loss='categorical_crossentropy', checkpoint=False):
        self.arch = arch
        self.params = params
        self.scale = scale
        self.loss = loss
        self.checkpoint = checkpoint


    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, do_class=True,**kwa):
        params = self.params

        if callable(params):
            params = params()

        np.random.seed(seed * 11 + 137)


        if self.scale:
            self.scaler = StandardScaler(with_mean=False)

            X_train = self.scaler.fit_transform(X_train)

            if X_eval is not None:
                X_eval = self.scaler.transform(X_eval)
        if do_class:
            y_train = to_categorical(y_train, num_classes=12)
            y_eval = y_eval if y_eval is None else to_categorical(y_eval, num_classes=12)


        checkpoint_path = "/tmp/nn-weights-%d.h5" % seed

        self.model = self.arch((X_train.shape[1],), params)
        self.model.compile(optimizer=params.get('optimizer', 'adadelta'), loss=self.loss,metrics=[f1])

        callbacks = list(params.get('callbacks', []))

        if self.checkpoint:
            callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0))

        self.model.fit_generator(
            generator=batch_generator(X_train, y_train, params['batch_size'], True), steps_per_epoch=X_train.shape[0]//params['batch_size'],
            validation_data=batch_generator(X_eval, y_eval, 800) if X_eval is not None else None, validation_steps=(X_eval.shape[0]//800)+1 if X_eval is not None else None,
            nb_epoch=params['n_epoch'], verbose=1, callbacks=callbacks)

        if self.checkpoint and os.path.isfile(checkpoint_path):
            self.model.load_weights(checkpoint_path)

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)

        see = self.model.predict_generator(batch_generator(X, batch_size=800), steps=X.shape[0] / 800 if X.shape[0] % 800 ==0 else (X.shape[0] // 800)+ 1,verbose=1)

        print 'see',see.shape

        return see.reshape((X.shape[0],-1))

def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1_l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None

def nn_lr(input_shape, params):
    model = Sequential()
    # model.add(Dense(1, input_shape=input_shape))
    model.add(Dense(units=12, init='he_normal', activation='softmax'))


    return model

def nn_mlp(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

        model.add(PReLU())

    # model.add(Dense(1, init='he_normal'))

    model.add(Dense(units=12,  activation='softmax'))
    print model.summary()

    return model

def nn_mlp_2(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        model.add(PReLU())

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    # model.add(Dense(1, init='he_normal'))
    model.add(Dense(units=12, init='he_normal', activation='softmax'))

    return model