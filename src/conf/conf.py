# -*- coding: utf-8 -*-
__author__ = 'lijingjie'
###########models##################################################
##1.xgb 2.et 3.rf 4.lgb 5.libfm
##6.lr 7.mlp 8.knn 9.svm 10.qr(QuantileRegression)
## goging adding ffm
####################################################################
import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/')
sys.path.insert(0, 'models')
sys.path.insert(0, './')


from deep_model import *
from meta_model import *
from traditional_model import *
import itertools
from scipy.stats import boxcox
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from keras.optimizers import SGD, Adam, Adadelta
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD

#############transform###############################################
###include boxcox log offset
###inverse above all
#####################################################################
norm_y_lambda = 0.7
def norm_y(y):
    return boxcox(np.log1p(y), lmbda=norm_y_lambda)
def norm_y_inv(y_bc):
    return np.expm1((y_bc * norm_y_lambda + 1)**(1/norm_y_lambda))

y_norm = (norm_y, norm_y_inv)
y_log = (np.log, np.exp)
def y_log_ofs(ofs):
    def transform(y):
        return np.log(y + ofs)

    def inv_transform(yl):
        return np.clip(np.exp(yl) - ofs, 1.0, np.inf)

    return transform, inv_transform

def y_pow(p):
    def transform(y):
        return y ** p

    def inv_transform(y):
        return y ** (1 / p)

    return transform, inv_transform

def y_pow_ofs(p, ofs):
    def transform(y):
        return (y + ofs) ** p

    def inv_transform(y):
        return np.clip(y ** (1 / p) - ofs, 1.0, np.inf)

    return transform, inv_transform

#for stage2 and stage3 outputs features
l1_predictions = [
    '20181204-2003-lr-ce-1278.84184',
    '20181204-2029-lr-cd-1237.43406',
    '20181204-2357-lr-cd-2-1256.45156',
    '20181204-2047-lr-svd-1237.79692',
    '20181204-2128-lr-cd-nr-1237.32174',
    '20181204-2048-lr-svd-clrbf-1210.53687',
    '20181204-2130-lr-svd-clrbf-2-1202.70592',
    '20181204-2359-lr-svd-clrbf-3-1212.10956',

    '20181207-0538-et-ce-1207.07344',
    '20181207-0601-et-ce-2-1204.68542',
    '20181207-0618-et-ce-3-1199.82233',
    '20181207-0030-et-ce-4-1194.75138',

    '20181207-0309-rf-ce-2-1193.61802',
    '20181206-2200-rf-ce-3-1190.27838',
    '20181207-0257-rf-ce-4-1186.23675',

    '20181207-1041-rf-ce-rot-1-1236.84994',

    '20181205-0006-gb-ce-1151.11060',

    '20181207-1643-knn1-1370.65015',
    '20181207-2025-knn2-1364.78537',

    '20181208-0022-svr1-1224.28418',

    '20181210-1914-nn-cd-2-1134.92794',
    '20181210-1651-nn-cd-3-1132.48048',
    '20181206-2201-nn-cd-4-1132.05160',

    '20181210-1142-nn-cd-clrbf-1132.71487',
    '20181207-1509-nn-cd-clrbf-2-1131.72969',
    '20181209-1459-nn-cd-clrbf-3-1132.49145',
    '20181127-2119-nn-cd-clrbf-4-1136.33283',#
    '20181209-0136-nn-cd-clrbf-5-1131.69357',
    '20181201-0719-nn-cd-clrbf-6-1145.02510',#
    '20181211-0636-nn-cd-clrbf-7-1133.32506',

    '20181207-0622-nn-svd-cd-clrbf-1-1130.29286',
    '20181201-2010-nn-svd-cd-clrbf-2-1135.79176',#
    '20181206-1656-nn-svd-cd-clrbf-3-1132.02805',

    '20181211-0413-lgb-cd-1-1133.89988',
    '20181207-2034-lgb-cd-2-1131.65831',

    '20181209-1932-lgb-ce-1-1129.07923',
    '20181206-1926-lgb-ce-2-1127.68636',

    '20181209-0939-xgb-ce-2-1133.00048',
    '20181208-1351-xgb-ce-3-1132.08820',

    '20181209-1442-xgbf-ce-2-1133.85036',
    '20181209-1109-xgbf-ce-3-1128.63753',
    '20181208-1109-xgbf-ce-4-1128.84209',
    '20181208-2307-xgbf-ce-4-2-1126.89842',
    '20181204-1303-xgbf-ce-5-1131.40964',
    '20181205-1313-xgbf-ce-6-1128.77616',
    '20181206-1400-xgbf-ce-7-1126.68014',
    '20181207-0946-xgbf-ce-8-1124.33319',
    '20181204-1025-xgbf-ce-9-1123.42983',
    '20181207-2128-xgbf-ce-10-1125.78132',
    '20181206-0327-xgbf-ce-12-1138.85463',
    '20181207-0954-xgbf-ce-13-1122.64977',
    '20181210-0733-xgbf-ce-14-1125.56181',

    '20181209-0336-xgbf-ce-clrbf-1-1151.51483',
    '20181204-2046-xgbf-ce-clrbf-2-1139.21753',

    '20181205-0123-libfm-cd-1196.11333',
    '20181205-1342-libfm-svd-1177.69251',
]
l2_predictions = [
    ([
        '20181209-2249-l2-knn-1128.69039',
        '20181209-2321-l2-svd-knn-1128.60543',

        '20181203-0232-l2-knn-1128.52203',
        '20181203-0135-l2-svd-knn-1128.44971',
        '20181130-0230-l2-svd-svr-1128.15513',
    ], {'power': 1.05}),

    ([
        '20181210-2219-l2-lr-1119.94373',
        '20181210-2219-l2-lr-2-1118.49848',
        '20181210-2220-l2-lr-3-1118.45564',
    ], {'power': 1.03}),

    [
        '20181205-0025-l2-qr-1117.03435',
        '20181207-0053-l2-qr-1116.97884',
        '20181209-1948-l2-qr-1116.63408',
    ],

    [
        '20181209-2101-l2-gb-1117.93768',
        '20181202-0020-l2-gb-1118.40560',
        '20181211-1858-l2-gb-1117.60834',
        '20181211-2153-l2-gb-2-1117.41247',
    ],

    ([
        '20181125-0753-l2-xgbf-1119.04996',
        '20181130-0258-l2-xgbf-1118.96658', #
        '20181202-0702-l2-xgbf-1118.63437', #
        '20181203-0636-l2-xgbf-1118.58470', #
        '20181210-0712-l2-xgbf-1118.33470',
    ], {'power': 1.02}),

    ([
        '20181202-1724-l2-xgbf-2-1118.43083',
        '20181210-1952-l2-xgbf-2-1118.15322',

        '20181130-2133-l2-xgbf-3-1118.75364', #
        '20181203-1336-l2-xgbf-3-1118.46005', #
        '20181211-0607-l2-xgbf-3-1118.16984',
    ], {'power': 1.02}),

    [
        '20181129-1219-l2-nn-1117.84214', #
        '20181202-0157-l2-nn-1117.33963', #
        '20181205-0337-l2-nn-1117.09224',
        '20181208-0126-l2-nn-1117.15231',
        '20181209-1814-l2-nn-1117.10987',
    ],

    [
        '20181124-1430-l2-nn-2-1117.28245', #
        '20181125-0958-l2-nn-2-1117.29028', #
        '20181129-1021-l2-nn-2-1117.39540', #
        '20181129-2228-l2-nn-2-1117.01003', #
        '20181202-0007-l2-nn-2-1116.65633', #
        '20181207-2324-l2-nn-2-1116.84297',
    ],

    [
        '20181210-0302-l2-nn-3-1116.53105',
        '20181212-1230-l2-nn-3-1116.40752',
        '20181208-1708-l2-nn-5-1116.86086',
        '20181210-0624-l2-nn-5-1116.83111',
        '20181211-1757-l2-nn-6-1116.60009',
    ],

    [
        '20181211-1956-l2-xgbf-4-1118.39030',
        '20181212-0519-l2-xgbf-4-2-1118.44814',
        '20181212-1552-l2-xgbf-4-3-1118.46351',
        '20181212-0959-l2-xgbf-5-1118.80387',
        '20181212-1950-l2-xgbf-5-2-1118.46393',
    ]
]
#configure for stage1 stage2 stage3
presets = {
    'xgb-tst': {
        'features': ['numeric','categorical','svd'],
        'model': Xgb({'objective': 'multi:softprob',
                      'eval_metric': 'mlogloss',
                      'num_class': 12,
                      'max_depth': 31,
                      'learning_rate': 0.05,
                      'num_leaves': 31,
                      'lambda_l1': 0.01,
                      'lambda_l2': 10,
                      'seed': 2019,
                      'feature_fraction': 0.8,
                      'bagging_fraction': 0.8,
                      'bagging_freq': 4,
                      }, n_iter=1),
        'param_grid': {'colsample_bytree': [0.2, 1.0]},
    },

    'xgb2': {
        'features': ['numeric', 'categorical_counts'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
        }, n_iter=400),
        'param_grid': {'colsample_bytree': [0.2, 1.0]},
    },

    'xgb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
            'gamma': 0.2,
        }, n_iter=2000),
    },

    'xgb-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1,
            'alpha': 1,
        }, n_iter=3000),
    },

    'xgb-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 14,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1.5,
            'alpha': 1,
        }, n_iter=3000),
    },

    'xgb-ce-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 14,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1.5,
            'alpha': 1,
        }, n_iter=3000),
    },

    'xgb4': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
        }, n_iter=3000),
    },

    'xgb6': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgb7': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(C=10000, noisy=False, loo=False)],
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbh-ce': {
        'features': ['numeric', 'categorical_encoded'],
        #'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, huber=100),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
            'alpha': 0.0005,
        }, n_iter=1100, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.45,
            'alpha': 0.0005,
        }, n_iter=1320, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.5,
            'alpha': 0.5,
        }, n_iter=4000, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-4': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 15,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.5,
        }, n_iter=5400, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-4-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 15,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 1.5,
            'alpha': 1.0,
        }, n_iter=5400, fair=1),
        'param_grid': {'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]},
    },

    'xgbf-ce-5': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 9,
            'eta': 0.01,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
        }, n_iter=6500, fair=150, fair_decay=0.0003),
    },

    'xgbf-ce-6': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_norm,
        'n_bags': 1,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.5,
        }, n_iter=2, fair=1),
    },

    'xgbf-ce-7': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 1.2,
            'alpha': 1.0,
        }, n_iter=5000, fair=1),
    },

    'xgbf-ce-8': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=16000, fair=1),
    },

    'xgbf-ce-9': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.1,
            'alpha': 0.95,
        }, n_iter=16000, fair=1),
    },

    'xgbf-ce-10': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 6,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.03,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'min_child_weight': 100
        }, n_iter=720, fair=0.7),
    },

    'xgbf-ce-11': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.04,
            'colsample_bytree': 0.2,
            'subsample': 0.75,
            'gamma': 2.0,
            'alpha': 2.0,
        }, n_iter=10000, fair=200),
    },

    'xgbf-ce-12': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=16000, fair=1),
    },

    'xgbf-ce-13': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'y_transform': y_pow(0.25),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.007,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 2.2,
            'alpha': 1.2,
        }, n_iter=8000, fair=1),
    },

    'xgbf-ce-14': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat101', 'cat80'), ('cat79', 'cat80'), ('cat12', 'cat80'), ('cat101', 'cat81'), ('cat12', 'cat81'), ('cat12', 'cat79'), ('cat57', 'cat79'), ('cat1', 'cat80'), ('cat101', 'cat79'), ('cat1', 'cat81')]
            )],
        'y_transform': y_pow_ofs(0.202, 5),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.3,
            'alpha': 0.6,
        }, n_iter=8000, fair=2.0),
    },

    'xgbf-ce-15': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'y_transform': y_pow(0.24),
        'n_bags': 6,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.005,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 2.2,
            'alpha': 1.2,
        }, n_iter=11000, fair=1),
    },

    'xgbf-ce-clrbf-1': {
        'features': ['numeric', 'categorical_encoded', 'cluster_rbf_200'],
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
        }, n_iter=1250, fair=150, fair_decay=0.001),
    },

    'xgbf-ce-clrbf-2': {
        'features': ['numeric', 'categorical_encoded', 'cluster_rbf_50'],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.01,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
            'lambda': 2.1
        }, n_iter=4400, fair=150),
    },

    'xgbf-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 3,
        'y_transform': y_norm,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.45,
            'alpha': 0.0005,
            #'lambda': 1.0,
        }, n_iter=1100, fair=1),
    },

    'xgbf-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalMeanEncoded(
                C=10000, noisy=True, noise_std=0.1, loo=False,
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'y_transform': y_norm,
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
            'alpha': 0.0005,
        }, n_iter=500, fair=1),
    },

    'lgb-tst': {
        'features': ['numeric','categorical'],
        'n_bags': 1,
        'model': LightGBM({
            'num_iterations': 5000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.2,
            'bagging_fraction': 0.3,
            'bagging_freq': 20,
            'metric_freq': 10,
            'metric': 'multi_logloss',
            'num_class':12,
            'application':'multiclass'
        }),
    },
    'lgb-tst1': {
        'features': ['numeric','categorical','svd'],
        'n_bags': 1,
        'model': Official_LightGBM({
            'objective': 'multiclass',
            'metrics': 'multiclass',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'lambda_l1': 0.01,
            'lambda_l2': 10,
            'num_class': 12,
            'seed': 2019,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 4
        }, n_iter=1000),
    },
    'lgb-cd-1': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_norm,
        'n_bags': 4,
        'model': LightGBM({
            'num_iterations': 2150,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10
        }),
    },

    'lgb-cd-2': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': LightGBM({
            'num_iterations': 2900,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10
        }),
    },

    'lgb-cd-tst': {
        'features': ['numeric', 'categorical_dummy'],
        'y_transform': y_log_ofs(200),
        'n_bags': 2,
        'model': LightGBM({
            'num_iterations': 5000,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10,
            'metric': 'l1',
        }),
    },

    'lgb-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(C=10000, noisy=True, noise_std=0.1, loo=False)],
        'y_transform': y_log_ofs(200),
        #'n_bags': 2,
        'model': LightGBM({
            'num_iterations': 4000,
            'learning_rate': 0.006,
            'num_leaves': 250,
            'min_data_in_leaf': 2,
            'feature_fraction': 0.25,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'metric_freq': 10
        }),
    },

    'lgb-ce-1': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 8,
        'model': LightGBM({
            'application': 'regression_fair',
            'num_iterations': 9350,
            'learning_rate': 0.003,
            'num_leaves': 250,
            'min_data_in_leaf': 2,
            'feature_fraction': 0.25,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'metric': 'l1',
            'metric_freq': 40
        }),
    },

    'lgb-ce-2': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_pow(0.25),
        'n_bags': 4,
        'model': LightGBM({
            'application': 'regression_fair',
            'num_iterations': 8000,
            'learning_rate': 0.005,
            'num_leaves': 250,
            'min_data_in_leaf': 2,
            'feature_fraction': 0.25,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'metric': 'l1',
            'metric_freq': 40
        }),
    },
    'libfm-softmax-tst': {
        'features': ['numeric','categorical','svd'],
        'model': LibFM_softmax(params={
        },n_iter=1),
    },

    'libfm-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_log,
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.0001,
            'iter': 200,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }),
    },

    'libfm-svd': {
        'features': ['svd'],
        'y_transform': y_log,
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.00005,
            'iter': 365,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }),
    },

    'libfm-svd-clrbf': {
        'features': ['svd', 'cluster_rbf_25'],
        'y_transform': y_log,
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.00007,
            'iter': 350,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }),
    },

    'nn-tst': {
        'features': ['numeric','categorical','svd'],
        'model': Keras(nn_mlp, {'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 100, 'batch_size': 128, 'layers': [300,100]}),
    },

    'nn1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Keras(nn_mlp, {'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 100, 'batch_size': 48, 'layers': [400, 200], 'dropouts': [0.4, 0.2]}),
    },

    'nn-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(nn_mlp, lambda: {'l1': 2e-5, 'l2': 2e-5, 'n_epoch': 40, 'batch_size': 128, 'layers': [400, 200, 100], 'dropouts': [0.5, 0.4, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-3': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-4': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_log_ofs(500),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_200'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf-2': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 70, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf-3': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-6, 'n_epoch': 60, 'batch_size': 128, 'layers': [400, 200, 80], 'dropouts': [0.4, 0.3, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf-4': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [400, 200, 100], 'dropouts': [0.5, 0.3, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf-5': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'y_transform': y_log_ofs(200),
        'n_bags': 8,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 70, 'batch_size': 128, 'layers': [400, 200, 70], 'dropouts': [0.5, 0.3, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf-6': {
        'features': ['numeric_unskew', 'numeric_edges', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100'],
        'y_transform': y_log_ofs(200),
        'n_bags': 6,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 80, 'batch_size': 128, 'layers': [350, 170, 70], 'dropouts': [0.6, 0.3, 0.15], 'batch_norm': True, 'optimizer': Adam(decay=1e-6), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-clrbf-7': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75', 'cluster_rbf_100'],
        'y_transform': y_log_ofs(200),
        'n_bags': 8,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 70], 'dropouts': [0.5, 0.3, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-svd': {
        'features': ['svd'],
        'n_bags': 2,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-svd-2': {
        'features': ['svd'],
        'y_transform': y_log_ofs(200),
        'n_bags': 2,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-7, 'l2': 1e-7, 'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.2, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-svd-cd-clrbf-1': {
        'features': ['svd', 'categorical_dummy', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75'],
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-6, 'n_epoch': 70, 'batch_size': 128, 'layers': [400, 150, 60], 'dropouts': [0.4, 0.25, 0.25], 'batch_norm': True, 'optimizer': Adam(decay=1e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-svd-cd-clrbf-2': {
        'features': ['svd', 'categorical_dummy', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75', 'cluster_rbf_100'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-5, 'n_epoch': 45, 'batch_size': 128, 'layers': [400, 200, 200], 'dropouts': [0.4, 0.4, 0.3], 'batch_norm': True, 'optimizer': SGD(3e-3, momentum=0.8, nesterov=True, decay=2e-4), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),  # Adam(decay=1e-6)
    },

    'nn-svd-cd-clrbf-3': {
        'features': ['svd', 'categorical_dummy', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75'],
        'y_transform': y_pow(0.25),
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-6, 'n_epoch': 60, 'batch_size': 128, 'layers': [400, 150, 60], 'dropouts': [0.4, 0.25, 0.25], 'batch_norm': True, 'optimizer': Adam(decay=1e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(1000)],
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=True),
    },

    'gb-tst': {
        'features': ['numeric'],
        'model': Sklearn(GradientBoostingClassifier(n_estimators=1, max_depth=7, max_features=0.2)),
        'param_grid': {'n_estimators': (1, 400), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)},
    },
    'gb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2)),
        'param_grid': {'n_estimators': (200, 400), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)},
    },

    'ab-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(AdaBoostRegressor(loss='linear', n_estimators=300)),
        'param_grid': {'n_estimators': (50, 400), 'learning_rate': (0.1, 1.0)},
    },

    'et-tst': {
        'features': ['numeric'],
        # 'y_transform': y_log,
        'model': Sklearn(ExtraTreesClassifier(2, max_features=0.2, n_jobs=-1)),

    },

    'et-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log,
        'model': Sklearn(ExtraTreesRegressor(200, max_features=0.2, n_jobs=-1)),
    },

    'et-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(ExtraTreesRegressor(200, max_features=0.2, n_jobs=-1)),
    },

    'et-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(ExtraTreesRegressor(50, max_features=0.8, min_samples_split=26, max_depth=23, n_jobs=-1)),
    },

    'et-ce-4': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(ExtraTreesRegressor(400, max_features=0.623,  max_depth=29, min_samples_leaf=4, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)},
    },

    'et-ce-5': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_pow(0.25),
        'model': Sklearn(ExtraTreesRegressor(400, max_features=0.623,  max_depth=29, min_samples_leaf=4, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)},
    },

    'rf-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(RandomForestRegressor(100, min_samples_split=16, max_features=0.3, max_depth=26, n_jobs=-1)),
    },

    'rf-ce-3': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(RandomForestRegressor(400, max_features=0.62, max_depth=39, min_samples_leaf=5, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)},
    },

    'rf-ce-4': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_pow(0.25),
        'model': Sklearn(RandomForestRegressor(400, max_features=0.62, max_depth=39, min_samples_leaf=5, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)},
    },

    'rf-ce-rot-1': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'y_transform': y_log_ofs(200),
        'n_bags': 20,
        'sample': 0.9,
        'feature_sample': 0.9,
        'svd': 50,
        'model': Sklearn(RandomForestRegressor(30, max_features=0.7, max_depth=39, min_samples_leaf=5, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)},
    },

    'lr-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Ridge(1e-3)),
        'param_grid': {'C': (1e-3, 1e3)},
    },

    'lr-cd-2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'y_transform': y_norm,
        'model': Sklearn(Ridge(1e-3)),
        'param_grid': {'C': (1e-3, 1e3)},
    },

    'lr-cm': {
        'features': ['numeric_scaled'],
        'feature_builders': [
            CategoricalMeanEncoded(
                C=10000, noisy=True, noise_std=0.01, loo=True,
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'y_transform': y_log,
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-cd-nr': {
        'features': ['numeric_rank_norm', 'categorical_dummy'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('lr', Ridge(1e-3))])),
    },

    'lr-svd': {
        'features': ['svd'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-svd-clrbf': {
        'features': ['svd', 'cluster_rbf_200'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-svd-clrbf-2': {
        'features': ['svd', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75', 'cluster_rbf_100', 'cluster_rbf_200'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-svd-clrbf-3': {
        'features': ['svd', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75', 'cluster_rbf_100', 'cluster_rbf_200'],
        'y_transform': y_norm,
        'model': Sklearn(Ridge(1e-3)),
    },

    'lr-svd-clrbf-4': {
        'features': ['svd', 'cluster_rbf_25', 'cluster_rbf_50', 'cluster_rbf_75', 'cluster_rbf_100', 'cluster_rbf_200'],
        'y_transform': y_log,
        'model': Sklearn(Ridge(1e-3)),
    },

    'knn1': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('est', KNeighborsRegressor(5, n_jobs=-1))])),
    },

    'knn2': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('est', KNeighborsRegressor(20, n_jobs=-1))])),
        'sample': 0.2,
        'n_bags': 4,
    },

    'knn3': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('est', KNeighborsRegressor(30, n_jobs=-1))])),
        'sample': 0.3,
        'n_bags': 4,
    },

    'knn4-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('est', KNeighborsRegressor(20, n_jobs=-1))])),
        'sample': 0.2,
        'feature_sample': 0.5,
        'svd': 30,
        'n_bags': 4,
    },

    'svr1': {
        'features': ['numeric', 'categorical_encoded'],
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('est', SVR())])),
        'sample': 0.05,
        'n_bags': 8,
    },

    'l2-nn': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [50], 'dropouts': [0.1], 'optimizer': SGD(1e-3, momentum=0.8, nesterov=True, decay=3e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-2': {
        'predictions': l1_predictions,
        'n_bags': 6,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 40, 'batch_size': 128, 'layers': [200, 50], 'dropouts': [0.15, 0.1], 'optimizer': SGD(1e-3, momentum=0.8, nesterov=True, decay=3e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-3': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 3e-6, 'l2': 3e-6, 'n_epoch': 70, 'batch_size': 128, 'layers': [200, 100], 'dropouts': [0.15, 0.15], 'optimizer': SGD(1e-4, momentum=0.9, nesterov=True, decay=3e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-4': {
        'predictions': l1_predictions,
        'n_bags': 6,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [200, 50], 'dropouts': [0.15, 0.1], 'optimizer': SGD(1e-4, momentum=0.8, nesterov=True, decay=1e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-5': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 3e-6, 'l2': 3e-6, 'n_epoch': 50, 'batch_size': 128, 'layers': [200, 100], 'dropouts': [0.15, 0.15], 'optimizer': SGD(1e-4, momentum=0.9, nesterov=True, decay=5e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-6': {
        'predictions': l1_predictions,
        'powers': [1.015, 1.03],
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 3e-6, 'l2': 3e-6, 'n_epoch': 50, 'batch_size': 128, 'layers': [200, 100], 'dropouts': [0.15, 0.15], 'optimizer': SGD(1e-4, momentum=0.9, nesterov=True, decay=5e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-7': {
        'predictions': l1_predictions,
        'powers': [1.02, 1.04],
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 3e-6, 'l2': 3e-6, 'n_epoch': 70, 'batch_size': 128, 'layers': [200, 100], 'dropouts': [0.15, 0.15], 'optimizer': SGD(1e-4, momentum=0.9, nesterov=True, decay=5e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-lr': {
        'predictions': l1_predictions,
        'prediction_transform': lambda x: np.log(x+200),
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Ridge()),
    },

    'l2-lr-2': {
        'predictions': l1_predictions,
        'prediction_transform': lambda x: x ** 0.25,
        'y_transform': y_pow(0.25),
        'model': Sklearn(Ridge()),
    },

    'l2-lr-3': {
        'predictions': l1_predictions,
        'prediction_transform': lambda x: (x+5) ** 0.2,
        'y_transform': y_pow_ofs(0.2, 5),
        'model': Sklearn(Ridge()),
    },

    'l2-xgbf': {
        'predictions': l1_predictions,
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 4,
            'eta': 0.0025,
            'colsample_bytree': 0.4,
            'subsample': 0.75,
            'min_child_weight': 6,
        }, n_iter=5000, fair=1.0),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-2': {
        'predictions': l1_predictions,
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 3,
            'eta': 0.0025,
            'colsample_bytree': 0.4,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 1.0,
        }, n_iter=6600, fair=1.0),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-3': {
        'features': ['manual'],
        'predictions': l1_predictions,
        'y_transform': y_log_ofs(200),
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 3,
            'eta': 0.005,
            'colsample_bytree': 0.4,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 0.5,
            'alpha': 0.4,
        }, n_iter=5000, fair=1.0),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-4': {
        'features': ['manual'],
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 3,
            'eta': 0.005,
            'colsample_bytree': 0.3,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 1.5,
            'alpha': 1.3,
        }, n_iter=5000, fair=150),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-4-2': {
        'features': ['manual'],
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 3,
            'eta': 0.005,
            'colsample_bytree': 0.3,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 1.5,
            'alpha': 1.3,
        }, n_iter=5000, fair=100),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-4-3': {
        'features': ['manual'],
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 3,
            'eta': 0.005,
            'colsample_bytree': 0.3,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 1.5,
            'alpha': 1.3,
        }, n_iter=5000, fair=200),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-5': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'predictions': l1_predictions,
        'n_bags': 7,
        'model': Xgb({
            'max_depth': 4,
            'eta': 0.003,
            'colsample_bytree': 0.4,
            'subsample': 0.55,
            'min_child_weight': 3,
            'lambda': 3.0,
            'alpha': 3.5,
        }, n_iter=5000, fair=150),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-xgbf-5-2': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 4,
            'eta': 0.003,
            'colsample_bytree': 0.6,
            'subsample': 0.8,
            'min_child_weight': 3,
            'lambda': 3.0,
            'alpha': 3.5,
        }, n_iter=5000, fair=120),
        'param_grid': {'max_depth': (3, 7), 'min_child_weight': (1, 20), 'lambda': (0, 2.0), 'alpha': (0, 2.0), 'subsample': (0.5, 1.0)},
    },

    'l2-et': {
        'predictions': l1_predictions,
        'y_transform': y_pow_ofs(0.2, 5),
        'model': Sklearn(ExtraTreesRegressor(100, max_depth=11, max_features=0.8, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (1, 40), 'max_features': (0.05, 0.8), 'max_depth': (3, 20)},
    },

    'l2-rf': {
        'predictions': l1_predictions,
        'y_transform': y_pow_ofs(0.2, 5),
        'model': Sklearn(RandomForestRegressor(100, max_depth=9, max_features=0.8, min_samples_leaf=23, n_jobs=-1)),
        'param_grid': {'min_samples_leaf': (1, 40), 'max_features': (0.05, 0.8), 'max_depth': (3, 20)},
    },

    'l2-gb': {
        'predictions': l1_predictions,
        'n_bags': 2,
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=425, learning_rate=0.05, subsample=0.65, min_samples_leaf=9, max_depth=5, max_features=0.35)),
        'param_grid': {'n_estimators': (200, 500), 'max_depth': (1, 8), 'max_features': (0.1, 0.8), 'min_samples_leaf': (1, 20), 'subsample': (0.5, 1.0), 'learning_rate': (0.01, 0.3)},
    },

    'l2-gb-2': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=425, learning_rate=0.04, subsample=0.65, min_samples_leaf=9, max_depth=5, max_features=0.35)),
        'param_grid': {'n_estimators': (200, 500), 'max_depth': (1, 8), 'max_features': (0.1, 0.8), 'min_samples_leaf': (1, 20), 'subsample': (0.5, 1.0), 'learning_rate': (0.01, 0.3)},
    },

    'l2-svd-svr': {
        'predictions': l1_predictions,
        'prediction_transform': lambda x: np.log(x+200),
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('svd', TruncatedSVD(10)), ('est', SVR())])),
        'sample': 0.1,
        'n_bags': 4,
    },

    'l2-knn': {
        'predictions': l1_predictions,
        'prediction_transform': lambda x: np.log(x+200),
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('est', KNeighborsRegressor(100, 'distance', n_jobs=-1))])),
        'sample': 0.2,
        'n_bags': 4,
    },

    'l2-svd-knn': {
        'predictions': l1_predictions,
        'prediction_transform': lambda x: np.log(x+200),
        'y_transform': y_log_ofs(200),
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('svd', TruncatedSVD(10)), ('est', KNeighborsRegressor(100, 'distance', n_jobs=-1))])),
        'sample': 0.95,
        'n_bags': 4,
    },

    'l2-qr': {
        'predictions': l1_predictions,
        'model': QuantileRegression(),
        'feature_sample': 0.7,
        'svd': 20,
        'n_bags': 4,
    },

    'l3-nn': {
        'predictions': l2_predictions,
        'n_bags': 4,
        'model': Keras(nn_lr, lambda: {'l2': 1e-5, 'n_epoch': 1, 'batch_size': 128, 'optimizer': SGD(lr=2.0, momentum=0.8, nesterov=True, decay=1e-4)}),
        'agg': np.mean,
    },

    'l3-qr': {
        'predictions': l2_predictions,
        'model': QuantileRegression(),
        'agg': np.mean,
    },

    'l3-qr-foldavg': {
        'predictions': l2_predictions,
        'predictions_mode': 'foldavg',
        'model': QuantileRegression(),
        'agg': np.mean,
        'sample': 0.8,
        'n_bags': 8,
    },

    'l3-qr-foldavg-power': {
        'predictions': l2_predictions,
        'predictions_mode': 'foldavg',
        'model': QuantileRegression(),
        'agg': np.mean,
        'sample': 0.8,
        'n_bags': 4,
        'powers': [1.02],
    },

    'l3-qr-2': {
        'predictions': l2_predictions,
        'model': QuantileRegression(),
        'agg': np.mean,
        'sample': 0.95,
        'n_bags': 4,
    },

    'l3-qr-3': {
        'predictions': l2_predictions,
        'model': QuantileRegression(),
        'agg': np.mean,
        'sample': 0.95,
        'feature_sample': 0.9,
        'svd': 10,
        'n_bags': 6,
    },

    'l3-qr-4': {
        'predictions': l2_predictions,
        'model': QuantileRegression(),
        'agg': np.mean,
        'sample': 0.95,
        'feature_sample': 0.9,
        'svd': 12,
        'powers': [1.023, 1.03],
        'n_bags': 6,
    },

    'l3-xgbf': {
        'predictions': l2_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 5,
            'eta': 0.005,
            'colsample_bytree': 0.3,
            'subsample': 0.55,
            'min_child_weight': 3,
        }, n_iter=5000, fair=50),
    },

    'l3-nn': {
        'predictions': l2_predictions,
        'n_bags': 4,
        'model': Keras(nn_lr, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'optimizer': SGD(3e-2, momentum=0.8, nesterov=True, decay=3e-5), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },
}

