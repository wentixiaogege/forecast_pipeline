# -*- coding: utf-8 -*-
__author__ = 'lijingjie'

import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/conf/')
sys.path.insert(0, '../conf/')
sys.path.insert(0, '../models')
sys.path.insert(0, '../')
from sklearn.model_selection import StratifiedKFold,train_test_split
from conf_kdd_regular import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['LIGHTGBM_EXEC'] = "/Users/jacklee/LightGBM/lightgbm"

#获取特征或者中间预测结果
def load_x(ds, preset):
    feature_parts = [Dataset.load_part(ds, part) for part in preset.get('features', [])]
    prediction_parts = [load_prediction(ds, p, mode=preset.get('predictions_mode', 'fulltrain')) for p in preset.get('predictions', [])]
    prediction_parts = [p.clip(lower=0.1).values.reshape((p.shape[0], 1)) for p in prediction_parts] # 拦截clip

    if 'prediction_transform' in preset:
        prediction_parts = list(map(preset['prediction_transform'], prediction_parts))  # 是否需要对上一次的预测做变换

    return hstack(feature_parts + prediction_parts)

def extract_feature_names(preset):
    x = []

    for part in preset.get('features', []):
        x += Dataset.get_part_features(part)

    lp = 1
    for pred in preset.get('predictions', []):
        if type(pred) is list:
            x.append('pred_%d' % lp)
            lp += 1
        else:
            x.append(pred)

    return x
def add_powers(x, feature_names, powers):
    res_feature_names = list(feature_names)
    res = [x]

    for p in powers:
        res.append(x ** p)

        for f in feature_names:
            res_feature_names.append("%s^%s" % (f, str(p)))

    return hstack(res), res_feature_names

from sklearn.metrics import f1_score

def eval_f(y_pred, train_data):
    y_true = train_data if type(train_data) ==np.ndarray else train_data.label if hasattr(train_data,"label") else train_data.get_label()
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    print ('weighted-f1-score', score, True) if hasattr(train_data,"label") else ('weighted-f1-score', score)
    return ('weighted-f1-score', score, True) if hasattr(train_data,"label") else ('weighted-f1-score', score)

## Main part
# import argparse
# parser = argparse.ArgumentParser(description='Train model')
# parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
# parser.add_argument('--optimize', action='store_true', help='optimize model params')
# parser.add_argument('--fold', type=int, help='specify fold')
# parser.add_argument('--threads', type=int, default=4, help='specify thread count')
#
# args = parser.parse_args()
#
# Xgb.default_params['nthread'] = args.threads
# LightGBM.default_params['num_threads'] = args.threads
Xgb.default_params['nthread'] = 8
LightGBM.default_params['num_threads']=12

n_folds = 2
n_classes=12
# set_train = 'knn3'
# set_train = 'xgb-tst'
# set_train='lr-cd'
set_train='lgb-tst1'
# set_train='lgb-cd-1'
# set_train='lgb-tst1'
# set_train='lgb-cd-2'
# set_train='libfm-softmax-tst'
# set_train='gb-tst'
# set_train='nn-tst'
# set_train='nn-cd'
# set_train='nn-cd-2'
# set_train='l2-lgbf'
# set_train = args.preset

print("Preset: %s" % set_train)
preset = presets[set_train]


feature_builders = preset.get('feature_builders', [])

n_splits = preset.get('n_splits', 1)# split几次，对每一个split都做kfold,不同的随机数
n_bags = preset.get('n_bags', 1)# bagging 几次，可以对样本和特征做sampling

y_aggregator = preset.get('agg', np.mean)# 变换方式
y_transform, y_inv_transform = preset.get('y_transform', (lambda y: y, lambda y: y))

print("Loading train data...")
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'target')
train_p = np.zeros((train_x.shape[0], n_splits * n_bags,n_classes))
train_r = Dataset.load('train', parts=np.unique(sum([b.requirements for b in feature_builders], ['target'])))# trian reconstruct features

feature_names = extract_feature_names(preset)

print 'using feature_name',feature_names
# if args.optimize:
#是否要优化参数
if False:
    opt_train_idx, opt_eval_idx = train_test_split(list(range(len(train_y))), test_size=0.2)

    opt_train_x = train_x[opt_train_idx]
    opt_train_y = train_y[opt_train_idx]
    opt_train_r = train_r.slice(opt_train_idx)

    opt_eval_x = train_x[opt_eval_idx]
    opt_eval_y = train_y[opt_eval_idx]
    opt_eval_r = train_r.slice(opt_eval_idx)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print("    Building per-fold features...")

        opt_train_x = [opt_train_x]
        opt_eval_x = [opt_eval_x]

        for fb in feature_builders:
            opt_train_x.append(fb.fit_transform(opt_train_r))
            opt_eval_x.append(fb.transform(opt_eval_r))

        opt_train_x = hstack(opt_train_x)
        opt_eval_x = hstack(opt_eval_x)

    preset['model'].optimize(opt_train_x, y_transform(opt_train_y), opt_eval_x, y_transform(opt_eval_y), preset['param_grid'], eval_func=eval_f)

print("Loading test data...")
test_x = load_x('test', preset)
test_r = Dataset.load('test', parts=np.unique([b.requirements for b in feature_builders]))
test_foldavg_p = np.zeros((test_x.shape[0], n_splits * n_bags * n_folds,n_classes))
test_fulltrain_p = np.zeros((test_x.shape[0], n_bags,n_classes))

#非线性
if 'powers' in preset:
    print("Adding power features...")

    train_x, feature_names = add_powers(train_x, feature_names, preset['powers'])
    test_x = add_powers(test_x, feature_names, preset['powers'])[0]

f1_scores = []
#按照splits fold去训练
if True:
    for split in range(n_splits):
        print("Training split %d..." % split)

        # for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2018 + 17*split)):
        skf = StratifiedKFold(n_splits=n_folds, random_state=2019 + 17*split, shuffle=True)

        for fold, (fold_train_idx, fold_eval_idx) in enumerate(skf.split(train_x,train_y)):
        #     print (n_folds)
        # for fold,fold_idx in enumerate(KFold(n_folds, shuffle=True, random_state=2019 + 17 * split).split(train_y)):


            # if args.fold is not None and fold != args.fold:
            #     continue
            # fold_train_idx = fold_idx[0]
            # fold_eval_idx = fold_idx[1]
            print(fold)
            print("  Fold %d..." % fold)

            fold_train_x = train_x[fold_train_idx]
            fold_train_y = train_y[fold_train_idx]
            fold_train_r = train_r.slice(fold_train_idx)

            fold_eval_x = train_x[fold_eval_idx]
            fold_eval_y = train_y[fold_eval_idx]
            fold_eval_r = train_r.slice(fold_eval_idx)

            fold_test_x = test_x
            fold_test_r = test_r

            fold_feature_names = list(feature_names)

            if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
                print("    Building per-fold features...")

                fold_train_x = [fold_train_x]
                fold_eval_x = [fold_eval_x]
                fold_test_x = [fold_test_x]

                # 这里去生成builder 特征
                for fb in feature_builders:
                    fold_train_x.append(fb.fit_transform(fold_train_r)) # 这里fb只是读取了categories特征，click_mode没管了
                    fold_eval_x.append(fb.transform(fold_eval_r))
                    fold_test_x.append(fb.transform(fold_test_r))
                    fold_feature_names += fb.get_feature_names()

                fold_train_x = hstack(fold_train_x)
                fold_eval_x = hstack(fold_eval_x)
                fold_test_x = hstack(fold_test_x)

            # eval_p = np.zeros((fold_eval_x.shape[0], n_bags)) # bag的结果比较
            eval_p = np.zeros((fold_eval_x.shape[0], n_bags,n_classes)) # bag的结果比较


            for bag in range(n_bags):
                print("    Training model %d..." % bag)

                rs = np.random.RandomState(101 + 31*split + 13*fold + 29*bag)

                bag_train_x = fold_train_x
                bag_train_y = fold_train_y

                bag_eval_x = fold_eval_x
                bag_eval_y = fold_eval_y

                bag_test_x = fold_test_x

                if 'sample' in preset:
                    bag_train_x, bag_train_y = resample(fold_train_x, fold_train_y, replace=False, n_samples=int(preset['sample'] * fold_train_x.shape[0]), random_state=42 + 11*split + 13*fold + 17*bag)

                if 'feature_sample' in preset:
                    features = rs.choice(list(range(bag_train_x.shape[1])), int(bag_train_x.shape[1] * preset['feature_sample']), replace=False)

                    bag_train_x = bag_train_x[:, features]
                    bag_eval_x = bag_eval_x[:, features]
                    bag_test_x = bag_test_x[:, features]

                if 'svd' in preset:
                    print "inside"
                    svd = TruncatedSVD(preset['svd'])

                    bag_train_x = svd.fit_transform(bag_train_x)
                    bag_eval_x = svd.transform(bag_eval_x)
                    bag_test_x = svd.transform(bag_test_x)




                pe, pt = preset['model'].fit_predict(train=(bag_train_x, y_transform(bag_train_y)),
                                                     val=(bag_eval_x, y_transform(bag_eval_y)),
                                                     test=(bag_test_x, ),
                                                     seed=42 + 11*split + 17*fold + 13*bag,
                                                     feature_names=fold_feature_names,
                                                     # eval_func=lambda yt, yp: log_loss(y_inv_transform(yt), y_inv_transform(yp)),
                                                     eval_func=eval_f,
                                                     name='%s-fold-%d-%d' % (set_train, fold, bag))

                eval_p[:, bag] += pe
                test_foldavg_p[:, split * n_folds * n_bags + fold * n_bags + bag] = pt # 记录下来所有的test预测结果

                train_p[fold_eval_idx, split * n_bags + bag] = pe #oof 预测

                print("    f1_score of model: %.5f" % f1_score(fold_eval_y, y_inv_transform(pe).argmax(-1), average='weighted'))

            print("  f1_score of mean-transform: %.5f" % f1_score(fold_eval_y, y_inv_transform(np.mean(eval_p, axis=1)).argmax(-1),average='weighted'))
            print("  f1_score of transform-mean: %.5f" % f1_score(fold_eval_y, np.mean(y_inv_transform(eval_p), axis=1).argmax(-1),average='weighted'))
            print("  f1_score of transform-median: %.5f" % f1_score(fold_eval_y, np.median(y_inv_transform(eval_p), axis=1).argmax(-1),average='weighted'))

            # Calculate err
            f1_scores.append(f1_score(fold_eval_y, y_aggregator(y_inv_transform(eval_p), axis=1).argmax(-1),average='weighted'))
            print("  f1_score: %.5f" % f1_scores[-1])

            # Free mem
            del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y

#全量去训练
if True:
    print("  Full...")

    full_train_x = train_x
    full_train_y = train_y
    full_train_r = train_r

    full_test_x = test_x
    full_test_r = test_r

    full_feature_names = list(feature_names)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print("    Building per-fold features...")

        full_train_x = [full_train_x]
        full_test_x = [full_test_x]

        for fb in feature_builders:
            full_train_x.append(fb.fit_transform(full_train_r))
            full_test_x.append(fb.transform(full_test_r))
            full_feature_names += fb.get_feature_names()

        full_train_x = hstack(full_train_x)
        full_test_x = hstack(full_test_x)

    for bag in range(n_bags):
        print("    Training model %d..." % bag)

        rs = np.random.RandomState(101 + 31*n_splits + 13*n_folds + 29*bag)

        bag_train_x = full_train_x
        bag_train_y = full_train_y

        bag_test_x = full_test_x

        if 'sample' in preset:
            bag_train_x, bag_train_y = resample(bag_train_x, bag_train_y, replace=False, n_samples=int(preset['sample'] * bag_train_x.shape[0]), random_state=42 + 11*split + 13*fold + 17*bag)

        if 'feature_sample' in preset:
            features = rs.choice(list(range(bag_train_x.shape[1])), int(bag_train_x.shape[1] * preset['feature_sample']), replace=False)

            bag_train_x = bag_train_x[:, features]
            bag_test_x = bag_test_x[:, features]

        if 'svd' in preset:
            svd = TruncatedSVD(preset['svd'])

            bag_train_x = svd.fit_transform(bag_train_x)
            bag_test_x = svd.transform(bag_test_x)

        fold_feature_names = list(feature_names)

        print ('preset model checking',preset['model'].n_iter)
        pt = preset['model'].fit_predict(train=(bag_train_x, y_transform(bag_train_y)),
                                         test=(bag_test_x, ),
                                         seed=42 + 11*n_splits + 17*n_folds + 13*bag,
                                         feature_names=fold_feature_names,
                                         eval_func=eval_f,
                                         size_mult=n_folds / (n_folds - 1.0),
                                         name='%s-full-%d' % (set_train, bag))

        test_fulltrain_p[:, bag] = pt

# Analyze predictions
f1_scores_mean = np.mean(f1_scores)
f1_scores_std = np.std(f1_scores)
f1_score1 = f1_score(train_y, y_aggregator(y_inv_transform(train_p), axis=1).argmax(-1),average='weighted')

# Aggregate predictions
name = "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), set_train, f1_score1)
test_foldavg_p_backup = pd.DataFrame(y_aggregator(y_inv_transform(test_foldavg_p), axis=1), index=Dataset.load_part('test', 'id'))
test_foldavg_p_backup.to_csv('preds/%s-%s.csv' % (name, 'test_foldavg_p_backup'), header=True)
test_fulltrain_p_backup = pd.DataFrame(y_aggregator(y_inv_transform(test_fulltrain_p), axis=1), index=Dataset.load_part('test', 'id'))
test_foldavg_p_backup.to_csv('preds/%s-%s.csv' % (name, 'test_foldavg_p_backup'), header=True)


train_p = pd.Series(np.argmax(y_aggregator(y_inv_transform(train_p), axis=1),axis=1), index=Dataset.load_part('train', 'id'))
test_foldavg_p = pd.Series(np.argmax(y_aggregator(y_inv_transform(test_foldavg_p), axis=1),axis=1), index=Dataset.load_part('test', 'id'))
test_fulltrain_p = pd.Series(np.argmax(y_aggregator(y_inv_transform(test_fulltrain_p), axis=1),axis=1), index=Dataset.load_part('test', 'id'))

print('---------------------------------')
print("CV f1_score: %.5f +- %.5f" % (f1_scores_mean, f1_scores_std))
print("CV RES f1_score: %.5f" % f1_score1)


print('---------------------------------')
print("Saving predictions... (%s)" % name)

for part, pred in [('train', train_p), ('test-foldavg', test_foldavg_p), ('test-fulltrain', test_fulltrain_p)]:
    pred.rename('click_mode', inplace=True)
    pred.index.rename('sid', inplace=True)
    pred.to_csv('preds/%s-%s.csv' % (name, part), header=True)

#copy current file
# copy2(os.path.realpath(__file__), os.path.join("preds", "%s-code.py" % name))

print("Done.")
