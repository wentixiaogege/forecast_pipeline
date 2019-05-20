# -*- coding: utf-8 -*-
__author__ = 'lijingjie'
import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/')
sys.path.insert(0, '../conf')
sys.path.insert(0, 'conf/')
import xgboost as xgb
import datetime
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle, resample
from statsmodels.regression.quantile_regression import QuantReg
from pylightgbm.models import GBMRegressor,GBMClassifier
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import os
import lightgbm as lgb

class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None,prob=True, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if prob:
            if val is None:
                return self.predict_proba(test[0])
            else:
                return self.predict_proba(val[0]),self.predict_proba(test[0])
        else:
            if val is None:
                return self.predict(test[0])
            else:
                return self.predict(val[0]), self.predict(test[0])

class Xgb(BaseAlgo):

    default_params = {
        'silent': 1,
        'nthread': -1
    }

    def __init__(self, params, n_iter=400, huber=None, fair=None, fair_decay=0):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.huber = huber
        self.fair = fair
        self.fair_decay = fair_decay
        self.tmp_dir = "xgboost/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M'))

        if self.huber is not None:
            self.objective = self.huber_approx_obj
        elif self.fair is not None:
            self.objective = self.fair_obj
        else:
            self.objective = None

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, size_mult=None, name=None):
        # feval = lambda y_pred, y_true: ('logloss', eval_func(y_true.get_label(), y_pred))

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

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

        self.model = xgb.train(params, dtrain, num_boost_round=n_iter, evals=watchlist,obj=self.objective, feval=eval_func,early_stopping_rounds=100, verbose_eval=2)
        self.model.dump_model('xgb-%s.dump' % name, with_stats=True)
        self.feature_names = feature_names

        print("      Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(list(self.model.get_fscore().items()), key=lambda t: -t[1])))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))
    def predict_proba(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func=None, seed=42):
        feval = lambda y_pred, y_true: ('logloss', eval_func(y_true.get_label(), y_pred))

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

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], self.objective, feval, verbose_eval=2, early_stopping_rounds=100)

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

class Official_LightGBM(BaseAlgo):
    default_params = {
        'silent': 0,
        'nthread': -1
    }

    def __init__(self, params, n_iter=4000, huber=None, fair=None, fair_decay=0):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        print ('ljj check lightgbm iters',self.n_iter)
        self.huber = huber
        self.fair = fair
        self.fair_decay = fair_decay
        self.tmp_dir = "lightgbm/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M'))


        if self.huber is not None:
            self.objective = self.huber_approx_obj
        elif self.fair is not None:
            self.objective = self.fair_obj
        else:
            self.objective = None

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None,
            size_mult=None, name=None):

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        # feval = lambda y_pred, y_true: ('weighted-f1-score', eval_func(y_pred,y_true))

        params = self.params.copy()
        params['seed'] = seed
        params['base_score'] = np.median(y_train)

        cate_featues = [i for i in feature_names if i.startswith('cat') ]
        print 'cate_features',cate_featues

        dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_names,categorical_feature=cate_featues,weight=[2 if i in [3,4,6,0] else 1 for i in y_train])

        if X_eval is None:
            watchlist = [dtrain]
            watchnames = ['train']
        else:
            deval = lgb.Dataset(X_eval, y_eval, feature_name=feature_names,categorical_feature=cate_featues)
            watchlist = [deval,dtrain]
            watchnames = ['eval','train']


        if size_mult is None:
            n_iter = self.n_iter
        else:
            n_iter = int(self.n_iter * size_mult)

        self.model = lgb.train(params, dtrain,num_boost_round=4000,valid_sets=watchlist,valid_names=watchnames,feval=eval_func,early_stopping_rounds=50, verbose_eval=20)
        self.model.save_model(os.path.join(self.tmp_dir, 'lgb-%s.dump' % name))
        self.feature_names = feature_names

        print("      Feature importances: %s" % ', '.join(
            '%s: %d' % t for t in sorted(list(zip(self.model.feature_name() ,self.model.feature_importance())), key=lambda t: -t[1])))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict(X)

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, feval=None, seed=42):
        feval = lambda y_pred, y_true: ('logloss', feval(y_true.get_label(), y_pred))

        dtrain = lgb.Dataset(X_train, label=y_train)
        deval = lgb.Dataset(X_eval, label=y_eval)

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

            model = lgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], self.objective, feval,
                              verbose_eval=2, early_stopping_rounds=100)

            print("Score %.5f at iteration %d" % (model.best_score, model.best_iteration))

            return - model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))

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

        self.model = GBMClassifier(**params)

        if X_eval is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, test_data=[(X_eval, y_eval)])

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
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

#https://github.com/CastellanZhang/alphaFM_softmax
class LibFM_softmax(BaseAlgo):

    default_params = {
    }

    def __init__(self, params={},n_iter=10):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.exec_path = '/Users/jacklee/alphaFM_softmax-master/bin/'
        self.tmp_dir = "libfm_models/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M'))
        self.first_train=True
        self.n_iter = n_iter

        # self.tmp_dir = "libfm_models/20190502-2156"

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
            dump_svmlight_file(*shuffle(X_train, 1+y_train, random_state=seed), f=f)

        if X_eval is None:
            eval_file = train_file
        else:
            print("Exporting eval...")
            with open(eval_file, 'w') as f:
                dump_svmlight_file(X_eval, 1+y_eval, f=f)

        params = self.params.copy()
        params['cn'] = 12
        params['core'] = 10
        params['m'] = os.path.join(self.tmp_dir, 'model.libfm_softmax')

        params_train = " ".join("-{} {}".format(k, params[k]) for k in params)

        for iter in np.arange(self.n_iter):

            if not self.first_train:
                print 'using pre-trained models'
                params['im'] = params['m']
                params.pop('out')if params.has_key('out') else None
                params_train = " ".join("-{} {}".format(k, params[k]) for k in params)

            command = "cat {} | {} {}".format(train_file,self.exec_path+'fm_train_softmax', params_train)

            print(command,iter)
            os.system(command)

            params['out'] = out_file
            params.pop('im') if params.has_key('im') else None
            params_eval = " ".join("-{} {}".format(k, params[k]) for k in params)

            eval_command = "cat {} | {} {}".format(eval_file,self.exec_path+'fm_predict_softmax', params_eval)

            print(eval_command,iter)
            os.system(eval_command)

            self.first_train = False


    def predict(self, X):
        # train_file = os.path.join(self.tmp_dir, 'train.svm')
        pred_file = os.path.join(self.tmp_dir, 'pred.svm')
        out_file = os.path.join(self.tmp_dir, 'pred_out.txt')
        os.system('rm -f ' + pred_file)
        os.system('rm -f ' + out_file)

        print("Exporting pred...")
        with open(pred_file, 'w') as f:
            dump_svmlight_file(X, 1+np.zeros(X.shape[0]), f=f)

        params = self.params.copy()
        # params['iter'] = 0
        params['cn'] = 12
        params['core'] = 10
        # params['train'] = train_file
        # params['test'] = pred_file
        params['out'] = out_file
        params['m'] = os.path.join(self.tmp_dir, 'model.libfm_softmax')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        pred_command = "cat {} | {} {}".format(pred_file, self.exec_path + 'fm_predict_softmax', params)

        print(pred_command)
        os.system(pred_command)

        # return pd.read_csv(out_file, header=None,sep=' ').values.flatten()
        return pd.read_csv(out_file,header=None,sep=' ').values[:,1:]
    def predict_proba(self, X):
        return self.predict(X)

class Sklearn(BaseAlgo):

    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        self.model.fit(X_train, y_train)

        if X_eval is not None and hasattr(self.model, 'staged_predict'):
            for i, p_eval in enumerate(self.model.staged_predict_proba(X_eval)):
                print("Iter %d score: %.5f" % (i, eval_func(p_eval,y_eval)[1]))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        print ('checking')
        return self.model.predict_proba(X)

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
