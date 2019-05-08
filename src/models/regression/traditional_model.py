# -*- coding: utf-8 -*-
__author__ = 'lijingjie'
import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/')
sys.path.insert(0, '../conf')
sys.path.insert(0, 'conf/')
import xgboost as xgb
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle, resample
from keras.callbacks import ModelCheckpoint
from statsmodels.regression.quantile_regression import QuantReg
from pylightgbm.models import GBMRegressor
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

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

class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            return self.predict(val[0]), self.predict(test[0])

class Xgb(BaseAlgo):

    default_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'nthread': -1,
    }

    def __init__(self, params, n_iter=400, huber=None, fair=None, fair_decay=0):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.huber = huber
        self.fair = fair
        self.fair_decay = fair_decay

        if self.huber is not None:
            self.objective = self.huber_approx_obj
        elif self.fair is not None:
            self.objective = self.fair_obj
        else:
            self.objective = None

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, size_mult=None, name=None):
        feval = lambda y_pred, y_true: ('mae', eval_func(y_true.get_label(), y_pred))

        params = self.params.copy()
        params['seed'] = seed
        params['base_score'] = np.median(y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

        if X_eval is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)
            watchlist = [(deval, 'eval'), (dtrain, 'train')]

        if size_mult is None:
            n_iter = self.n_iter
        else:
            n_iter = int(self.n_iter * size_mult)

        self.iter = 0
        self.model = xgb.train(params, dtrain, n_iter, watchlist, self.objective, feval, verbose_eval=20)
        # self.model.dump_model('xgb-%s.dump' % name, with_stats=True)
        self.feature_names = feature_names

        print("      Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(list(self.model.get_fscore().items()), key=lambda t: -t[1])))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func=None, seed=42):
        feval = lambda y_pred, y_true: ('mae', eval_func(y_true.get_label(), y_pred))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        def fun(**kw):
            params = self.params.copy()
            params['seed'] = seed
            params['base_score'] = np.median(y_train)

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print("Trying %s..." % str(params))

            self.iter = 0

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], self.objective, feval, verbose_eval=20, early_stopping_rounds=100)

            print("Score %.5f at iteration %d" % (model.best_score, model.best_iteration))

            return - model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))

    def huber_approx_obj(self, preds, dtrain):
        d = preds - dtrain.get_label()
        h = self.huber

        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)

        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt

        return grad, hess

    def fair_obj(self, preds, dtrain):
        x = preds - dtrain.get_label()
        c = self.fair

        den = np.abs(x) * np.exp(self.fair_decay * self.iter) + c

        grad = c*x / den
        hess = c*c / den ** 2

        self.iter += 1

        return grad, hess

class LightGBM(BaseAlgo):

    default_params = {
        'exec_path': 'lightgbm',
        'num_threads': 4
    }

    def __init__(self, params):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        params = self.params.copy()
        params['bagging_seed'] = seed
        params['feature_fraction_seed'] = seed + 3

        self.model = GBMRegressor(**params)

        if X_eval is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, test_data=[(X_eval, y_eval)])

    def predict(self, X):
        return self.model.predict(X)

class LibFM(BaseAlgo):

    default_params = {
    }

    def __init__(self, params={}):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.exec_path = 'libFM'
        self.tmp_dir = "libfm_models/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M'))

    def __del__(self):
        #if os.path.exists(self.tmp_dir):
        #    rmtree(self.tmp_dir)
        pass

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        train_file = os.path.join(self.tmp_dir, 'train.svm')
        eval_file = os.path.join(self.tmp_dir, 'eval.svm')
        out_file = os.path.join(self.tmp_dir, 'out.txt')

        print("Exporting train...")
        with open(train_file, 'w') as f:
            dump_svmlight_file(*shuffle(X_train, y_train, random_state=seed), f=f)

        if X_eval is None:
            eval_file = train_file
        else:
            print("Exporting eval...")
            with open(eval_file, 'w') as f:
                dump_svmlight_file(X_eval, y_eval, f=f)

        params = self.params.copy()
        params['seed'] = seed
        params['task'] = 'r'
        params['train'] = train_file
        params['test'] = eval_file
        params['out'] = out_file
        params['save_model'] = os.path.join(self.tmp_dir, 'model.libfm')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        command = "{} {}".format(self.exec_path, params)

        print(command)
        os.system(command)

    def predict(self, X):
        train_file = os.path.join(self.tmp_dir, 'train.svm')
        pred_file = os.path.join(self.tmp_dir, 'pred.svm')
        out_file = os.path.join(self.tmp_dir, 'out.txt')

        print("Exporting pred...")
        with open(pred_file, 'w') as f:
            dump_svmlight_file(X, np.zeros(X.shape[0]), f=f)

        params = self.params.copy()
        params['iter'] = 0
        params['task'] = 'r'
        params['train'] = train_file
        params['test'] = pred_file
        params['out'] = out_file
        params['load_model'] = os.path.join(self.tmp_dir, 'model.libfm')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        command = "{} {}".format(self.exec_path, params)

        print(command)
        os.system(command)

        return pd.read_csv(out_file, header=None).values.flatten()

class Sklearn(BaseAlgo):

    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        self.model.fit(X_train, y_train)

        if X_eval is not None and hasattr(self.model, 'staged_predict'):
            for i, p_eval in enumerate(self.model.staged_predict(X_eval)):
                print("Iter %d score: %.5f" % (i, eval_func(y_eval, p_eval)))

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func, seed=42):
        def fun(**params):
            for k in params:
                if type(param_grid[k][0]) is int:
                    params[k] = int(params[k])

            print("Trying %s..." % str(params))

            self.model.set_params(**params)
            self.fit(X_train, y_train)

            if hasattr(self.model, 'staged_predict'):
                best_score = 1e9
                best_i = -1
                for i, p_eval in enumerate(self.model.staged_predict(X_eval)):
                    mae = eval_func(y_eval, p_eval)

                    if mae < best_score:
                        best_score = mae
                        best_i = i

                print("Best score after %d iters: %.5f" % (best_i, best_score))
            else:
                p_eval = self.predict(X_eval)
                best_score = eval_func(y_eval, p_eval)

                print("Score: %.5f" % best_score)

            return -best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))

class QuantileRegression(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        model = QuantReg(train[1], train[0]).fit(q=0.5, max_iter=10000)

        if val is None:
            return model.predict(test[0])
        else:
            return model.predict(val[0]), model.predict(test[0])

class Keras(BaseAlgo):

    def __init__(self, arch, params, scale=True, loss='mae', checkpoint=False):
        self.arch = arch
        self.params = params
        self.scale = scale
        self.loss = loss
        self.checkpoint = checkpoint

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        params = self.params

        if callable(params):
            params = params()

        np.random.seed(seed * 11 + 137)

        if self.scale:
            self.scaler = StandardScaler(with_mean=False)

            X_train = self.scaler.fit_transform(X_train)

            if X_eval is not None:
                X_eval = self.scaler.transform(X_eval)

        checkpoint_path = "/tmp/nn-weights-%d.h5" % seed

        self.model = self.arch((X_train.shape[1],), params)
        self.model.compile(optimizer=params.get('optimizer', 'adadelta'), loss=self.loss)

        callbacks = list(params.get('callbacks', []))

        if self.checkpoint:
            callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0))

        self.model.fit_generator(
            generator=batch_generator(X_train, y_train, params['batch_size'], True), samples_per_epoch=X_train.shape[0],
            validation_data=batch_generator(X_eval, y_eval, 800) if X_eval is not None else None, nb_val_samples=X_eval.shape[0] if X_eval is not None else None,
            nb_epoch=params['n_epoch'], verbose=1, callbacks=callbacks)

        if self.checkpoint and os.path.isfile(checkpoint_path):
            self.model.load_weights(checkpoint_path)

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)

        return self.model.predict_generator(batch_generator(X, batch_size=800), val_samples=X.shape[0]).reshape((X.shape[0],))