# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm import tqdm
import json
def read_profile_data():
    profile_data = pd.read_csv('../../input/kdd2019_regular/phase1/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data

def gen_profile_feas(data):
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_profile_fea_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1)
    data = data.merge(svd_feas, on='pid', how='left')
    return data

def gen_od_feas(data):
    data['num_o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['num_o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['num_d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['num_d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    data = data.drop(['o', 'd'], axis=1)
    return data

def gen_plan_feas(data):
    n = data.shape[0]
    mode_list_feas = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    mode_texts = []
    for i, plan in (enumerate(data['plans'].values)):
        try:
            cur_plan_list = json.loads(plan)
        except:
            cur_plan_list = []
        if len(cur_plan_list) == 0:
            mode_list_feas[i, 0] = 1
            first_mode[i] = 0

            max_dist[i] = -1
            min_dist[i] = -1
            mean_dist[i] = -1
            std_dist[i] = -1

            max_price[i] = -1
            min_price[i] = -1
            mean_price[i] = -1
            std_price[i] = -1

            max_eta[i] = -1
            min_eta[i] = -1
            mean_eta[i] = -1
            std_eta[i] = -1

            min_dist_mode[i] = -1
            max_dist_mode[i] = -1
            min_price_mode[i] = -1
            max_price_mode[i] = -1
            min_eta_mode[i] = -1
            max_eta_mode[i] = -1

            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            for tmp_dit in cur_plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            mode_list = np.array(mode_list, dtype='int')
            mode_list_feas[i, mode_list] = 1
            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)

            max_dist[i] = distance_list[distance_sort_idx[-1]]
            min_dist[i] = distance_list[distance_sort_idx[0]]
            mean_dist[i] = np.mean(distance_list)
            std_dist[i] = np.std(distance_list)

            max_price[i] = price_list[price_sort_idx[-1]]
            min_price[i] = price_list[price_sort_idx[0]]
            mean_price[i] = np.mean(price_list)
            std_price[i] = np.std(price_list)

            max_eta[i] = eta_list[eta_sort_idx[-1]]
            min_eta[i] = eta_list[eta_sort_idx[0]]
            mean_eta[i] = np.mean(eta_list)
            std_eta[i] = np.std(eta_list)

            first_mode[i] = mode_list[0]
            max_dist_mode[i] = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i] = mode_list[distance_sort_idx[0]]

            max_price_mode[i] = mode_list[price_sort_idx[-1]]
            min_price_mode[i] = mode_list[price_sort_idx[0]]

            max_eta_mode[i] = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i] = mode_list[eta_sort_idx[0]]

    feature_data = pd.DataFrame(mode_list_feas)
    feature_data.columns = ['cat_mode_feas_{}'.format(i) for i in range(12)]
    feature_data['num_max_dist'] = max_dist
    feature_data['num_min_dist'] = min_dist
    feature_data['num_mean_dist'] = mean_dist
    feature_data['num_std_dist'] = std_dist

    feature_data['num_max_price'] = max_price
    feature_data['num_min_price'] = min_price
    feature_data['num_mean_price'] = mean_price
    feature_data['num_std_price'] = std_price

    feature_data['num_max_eta'] = max_eta
    feature_data['num_min_eta'] = min_eta
    feature_data['num_mean_eta'] = mean_eta
    feature_data['num_std_eta'] = std_eta

    feature_data['num_max_dist_mode'] = max_dist_mode
    feature_data['num_min_dist_mode'] = min_dist_mode
    feature_data['num_max_price_mode'] = max_price_mode
    feature_data['num_min_price_mode'] = min_price_mode
    feature_data['num_max_eta_mode'] = max_eta_mode
    feature_data['num_min_eta_mode'] = min_eta_mode
    feature_data['num_first_mode'] = first_mode
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_plan_mode_{}'.format(i) for i in range(10)]

    data = pd.concat([data, feature_data, mode_svd], axis=1)
    data = data.drop(['plans'], axis=1)
    return data

def gen_time_feas(data):
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['cat_weekday'] = data['req_time'].dt.dayofweek
    data['cat_is_weekday'] = data['req_time'].dt.dayofweek.apply(lambda x:0 if x in[1,2,3,4,5] else 1)
    data['cat_month']=data['req_time'].dt.month
    data['cat_season'] = data['req_time'].dt.month.apply(lambda x:0 if x<=3 else 1 if x <=6 else 2 if x<=9 else 3)
    data['cat_dayofyear'] = data['req_time'].dt.dayofyear
    data['cat_timeofday'] = data['req_time'].dt.hour.apply(lambda x: 0 if x<=6 else 1 if x<=12 else 2 if x<=18 else 3)
    # # data['cat_holiday'] =
    data['cat_hour'] = data['req_time'].dt.hour
    data = data.drop(['req_time'], axis=1)
    return data

def merge_raw_data():
    tr_queries = pd.read_csv('../../input/kdd2019_regular/phase1/train_queries.csv')
    print tr_queries.columns
    te_queries = pd.read_csv('../../input/kdd2019_regular/phase1/test_queries.csv')
    print te_queries.columns
    tr_plans = pd.read_csv('../../input/kdd2019_regular/phase1/train_plans.csv')
    print tr_plans.columns
    te_plans = pd.read_csv('../../input/kdd2019_regular/phase1/test_plans.csv')
    print te_plans.columns

    tr_click = pd.read_csv('../../input/kdd2019_regular/phase1/train_clicks.csv')
    print tr_click.columns


    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)
    te_data = te_queries.merge(te_plans, on='sid', how='left')
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)

    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data

def split_train_val(data):
    X = data[data.columns.difference(['click_mode'],sort=False)].values
    y = data[['click_mode']].values
    from sklearn.model_selection import train_test_split

    print X.shape
    print y.shape
    print data.shape
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.22, random_state=42)

    print X_train.shape
    print  y_train.shape
    print y_train.T.shape
    train = pd.DataFrame(np.concatenate((y_train,X_train), axis=1),columns=data.columns)
    val = pd.DataFrame(np.concatenate((y_val,X_val), axis=1),columns=data.columns)
    return train, val

if False:
    data = merge_raw_data()
    data = gen_od_feas(data)
    data = gen_profile_feas(data)
    data = gen_plan_feas(data)
    data = gen_time_feas(data)

    data.to_csv('../../input/kdd2019_regular/phase1/data.csv')
    # data=pd.read_csv('../../input/kdd2019_regular/phase1/data.csv', index_col=0)


    data[data.click_mode!=-1].to_csv('../../input/kdd2019_regular/phase1/train.csv')

    train,val = split_train_val(data[data.click_mode!=-1])
    train.to_csv('../../input/kdd2019_regular/phase1/train1.csv')
    val.to_csv('../../input/kdd2019_regular/phase1/val1.csv')

    data[data.click_mode==-1].to_csv('../../input/kdd2019_regular/phase1/test.csv')

for name in ['train','test','train1', 'val1']:
    print("Processing %s..." % name)
    data = pd.read_csv('../../input/kdd2019_regular/phase1/%s.csv' % name, index_col=0)

    # Save column names
    if name in ['train1','train']:
        cat_columns = [c for c in data.columns if c.startswith('cat')]
        num_columns = [c for c in data.columns if c.startswith('num')]

        Dataset.save_part_features('categorical', cat_columns)
        Dataset.save_part_features('numeric', num_columns)

        svd_columns = [c for c in data.columns if c.startswith('svd')]
        Dataset.save_part_features('svd', svd_columns)

    Dataset(categorical=data[cat_columns].values).save(name)
    Dataset(numeric=data[num_columns].values.astype(np.float32)).save(name)
    Dataset(sid=data['sid']).save(name)

    Dataset(svd=data[svd_columns].values).save(name)

    if 'click_mode' in data.columns:
        Dataset(click_mode=data['click_mode']).save(name)

print("Done.")
