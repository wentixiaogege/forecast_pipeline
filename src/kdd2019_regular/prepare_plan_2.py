# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def gen_plan_feas_origin(data):
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
    feature_data['cat_first_mode'] = first_mode
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



print("Loading data...")

data = Dataset.load_part('data','manual')
feature = Dataset.get_part_features('manual_data')

data_df = pd.DataFrame(data,columns=feature)

result = gen_plan_feas_origin(data_df)

cat_columns = [c for c in result.columns if c.startswith('cat')]
num_columns = [c for c in result.columns if c.startswith('num')]
svd_columns = [c for c in result.columns if c.startswith('svd')]
print('cat_columns',cat_columns)
print('num_columns',num_columns)

Dataset.save_part_features('categorical_plan', cat_columns)
Dataset.save_part_features('numeric_plan', num_columns)
Dataset.save_part_features('svd_plan', svd_columns)


Dataset(categorical=result[cat_columns].values).save('plan')
Dataset(numeric=result[num_columns].values).save('plan')
Dataset(svd=result[svd_columns].values).save('plan')

print('Done!')
