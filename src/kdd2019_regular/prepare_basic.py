# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import Dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm import tqdm
import json
from geopy.distance import geodesic


def read_profile_data():
    profile_data = pd.read_csv('../../input/kdd2019_regular/phase1/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1  # 增加pid=-1的
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    profile_data.columns = ['cat_' + i for i in profile_data.columns]
    return profile_data


def gen_profile_feas(data):
    profile_data = read_profile_data()
    print profile_data.head()
    x = profile_data.drop(['cat_pid'], axis=1).values
    svd = TruncatedSVD(n_components=10, n_iter=30, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_profile_fea_{}'.format(i) for i in range(10)]
    svd_feas['cat_pid'] = profile_data['cat_pid'].values
    data['cat_pid'] = data['cat_pid'].fillna(-1) # nan的pid 搞成了-1
    data = data.merge(svd_feas, on='cat_pid', how='left')
    limit_profile_data = profile_data[['cat_pid','cat_p13','cat_p29','cat_p33','cat_p9','cat_p6','cat_p5','cat_p0']] # 这些feature对0类别应该会有好的效果
    data = data.merge(limit_profile_data, on='cat_pid', how='left') # ---> adding origin pid features
    del profile_data
    del limit_profile_data
    del svd
    return data


def gen_od_feas(data):
    data['num_o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['num_o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['num_d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['num_d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))

    locations = pd.read_csv('../../input/kdd2019_regular/phase1/lntlat_adress_6525.csv')
    location_columns = ['adcode', 'district', 'lntlat', 'street', 'street_number']
    locations = locations[location_columns]

    data['cat_source_lntlat'] = data.o
    data['cat_des_lntlat'] = data.d

    locations.columns = map(lambda x: 'cat_source_' + x, location_columns)
    merge = pd.merge(data, locations, on=['cat_source_lntlat'], how='inner')

    locations.columns = map(lambda x: 'cat_des_' + x, location_columns)

    merge = pd.merge(merge, locations, on=['cat_des_lntlat'], how='inner')

    merge['cat_same_district'] = 0
    merge['cat_same_street_number'] = 0
    merge['cat_same_adcode'] = 0
    merge['cat_same_address'] = 0

    merge.loc[merge['cat_source_district'] == merge['cat_des_district'], 'cat_same_district'] = 1
    merge.loc[merge['cat_source_street_number'] == merge['cat_des_street_number'], 'cat_same_street_number'] = 1
    merge.loc[merge['cat_source_adcode'] == merge['cat_des_adcode'], 'cat_same_adcode'] = 1
    merge.loc[merge['o'] == merge['d'], 'cat_same_address'] = 1  # 大部分是不行 mode5 或者是0
    merge['num_direct_distance'] = merge.apply(lambda x: geodesic((x.num_o2, x.num_o1), (x.num_d2, x.num_d1)).m, axis=1)

    #
    from sklearn.preprocessing import LabelEncoder
    merge[['cat_source_district', 'cat_des_district']] = merge[['cat_source_district', 'cat_des_district']].apply(
        LabelEncoder().fit_transform)

    merge = merge.drop(
        ['o', 'd', 'cat_source_lntlat', 'cat_des_lntlat', 'cat_source_street', 'cat_source_street_number',
         'cat_des_street', 'cat_des_street_number'], axis=1)
    return merge


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

# def gen_plan_feas_by_plans(plans_df):
#     """
#
#     :param data:
#     :return:
#     """
#     mode_columns_names = ['cat_mode_feas_{}'.format(i) for i in range(12)]
#
#     def gen_mode_code(mode_list):
#         ma = np.zeros(12)
#         ma[mode_list] = 1
#         return ma
#
#     # 生成 一组计划的mode 占位符
#     mode_g = plans_df.groupby('sid')['transport_mode'].apply(gen_mode_code).reset_index()
#     mode_columns = ['sid'] + mode_columns_names
#     mode_data = np.concatenate(mode_g['transport_mode'].values, axis=0).reshape(len(mode_g), 12)
#     sid_data = mode_g['sid'].values.reshape(len(mode_g), 1)
#     mode_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)
#
#     def get_first(x):
#         return x.values[0]
#
#     def gen_mode_texts(x):
#         tl = ' '.join(['word_{}'.format(mode) for mode in x.values])
#         return tl
#
#     agg_fun = {'transport_mode': [get_first, gen_mode_texts],
#                'distance': ['max', 'min', 'mean', lambda x: np.std(x)],
#                'price': ['max', 'min', 'mean', lambda x: np.std(x)],
#                'eta': ['max', 'min', 'mean', lambda x: np.std(x)],
#                # 添加三组特征
#                'dj': ['max', 'min', 'mean', lambda x: np.std(x)],
#                'sd': ['max', 'min', 'mean', lambda x: np.std(x)],
#                'sd_dj': ['max', 'min', 'mean', lambda x: np.std(x)]
#                }
#     # std ddof =1
#     # [u'distance', u'dj', u'transport_mode', u'price', u'eta', u'sd_dj', u'sd', u'sid']
#     agg_columns = ['sid',
#                    'num_max_dist', 'num_min_dist', 'num_mean_dist', 'num_std_dist',
#                    'num_max_dj', 'num_min_dj', 'num_mean_dj', 'num_std_dj',
#                    'cat_first_mode', 'mode_texts',
#                    'num_max_price', 'num_min_price', 'num_mean_price', 'num_std_price',
#                    'num_max_eta', 'num_min_eta', 'num_mean_eta', 'num_std_eta',
#                    'num_max_sd_dj', 'num_min_sd_dj', 'num_mean_sd_dj', 'num_std_sd_dj',
#                    'num_max_sd', 'num_min_sd', 'num_mean_sd','num_std_sd'
#                    ]
#
#     agg_df = plans_df.groupby('sid').agg(agg_fun).reset_index()
#     print agg_df.head()
#     print agg_df.columns
#     agg_df.columns = agg_columns
#     merge_df = pd.merge(plans_df, agg_df, on=['sid'], how='inner')
#     # 原来版本是 keep='last'
#     max_dist_mode_df = merge_df.loc[merge_df['distance'] == merge_df['num_max_dist'], ['sid', 'transport_mode']]
#     max_dist_mode_df.columns = ['sid', 'cat_max_dist_mode']
#     max_dist_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
#     min_dist_mode_df = merge_df.loc[merge_df['distance'] == merge_df['num_min_dist'], ['sid', 'transport_mode']]
#     min_dist_mode_df.columns = ['sid', 'cat_min_dist_mode']
#     min_dist_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
#     max_price_mode_df = merge_df.loc[merge_df['price'] == merge_df['num_max_price'], ['sid', 'transport_mode']]
#     max_price_mode_df.columns = ['sid', 'cat_max_price_mode']
#     max_price_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
#     min_price_mode_df = merge_df.loc[merge_df['price'] == merge_df['num_min_price'], ['sid', 'transport_mode']]
#     min_price_mode_df.columns = ['sid', 'cat_min_price_mode']
#     min_price_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
#     max_eta_mode_df = merge_df.loc[merge_df['eta'] == merge_df['num_max_eta'], ['sid', 'transport_mode']]
#     max_eta_mode_df.columns = ['sid', 'cat_max_eta_mode']
#     max_eta_mode_df.drop_duplicates(subset='sid', keep='last', inplace=True)
#     min_eta_mode_df = merge_df.loc[merge_df['eta'] == merge_df['num_min_eta'], ['sid', 'transport_mode']]
#     min_eta_mode_df.columns = ['sid', 'cat_min_eta_mode']
#     min_eta_mode_df.drop_duplicates(subset='sid', keep='first', inplace=True)
#
#     complex_feature_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['sid'], how='inner'),
#                                 [max_dist_mode_df, min_dist_mode_df, max_price_mode_df, min_price_mode_df,
#                                  max_eta_mode_df, min_eta_mode_df])
#     plan_feature_df = reduce(lambda ldf, rdf: pd.merge(ldf, rdf, on=['sid'], how='inner'),
#                              [mode_df, agg_df, complex_feature_df])
#
#     return plan_feature_df


# def gen_empty_plan_feas(data):
#     """
#
#     生成empty plans
#     :param data:
#     :return:
#     """
#     mode_columns_names = ['cat_mode_feas_{}'.format(i) for i in range(12)]
#
#     mode_data = np.zeros((len(data), 12))
#     mode_data[:, 0] = 1
#     sid_data = data['sid'].values.reshape(len(data), 1)
#     mode_columns = ['sid'] + mode_columns_names
#     plan_feature_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)
#
#     plan_feature_df['cat_first_mode'] = 0
#     plan_feature_df['mode_texts'] = 'word_null'
#
#     plan_feature_df['num_max_dist'] = -1
#     plan_feature_df['num_min_dist'] = -1
#     plan_feature_df['num_mean_dist'] = -1
#     plan_feature_df['num_std_dist'] = -1
#
#     plan_feature_df['num_max_price'] = -1
#     plan_feature_df['num_min_price'] = -1
#     plan_feature_df['num_mean_price'] = -1
#     plan_feature_df['num_std_price'] = -1
#
#     plan_feature_df['num_max_eta'] = -1
#     plan_feature_df['num_min_eta'] = -1
#     plan_feature_df['num_mean_eta'] = -1
#     plan_feature_df['num_std_eta'] = -1
#
#     # 新增特征
#     plan_feature_df['num_max_dj'] = -1
#     plan_feature_df['num_min_dj'] = -1
#     plan_feature_df['num_mean_dj'] = -1
#     plan_feature_df['num_std_dj'] = -1
#
#     plan_feature_df['num_max_sd'] = -1
#     plan_feature_df['num_min_sd'] = -1
#     plan_feature_df['num_mean_sd'] = -1
#     plan_feature_df['num_std_sd'] = -1
#
#     plan_feature_df['num_max_sd_dj'] = -1
#     plan_feature_df['num_min_sd_dj'] = -1
#     plan_feature_df['num_mean_sd_dj'] = -1
#     plan_feature_df['num_std_sd_dj'] = -1
#
#     plan_feature_df['cat_max_dist_mode'] = -1
#     plan_feature_df['cat_min_dist_mode'] = -1
#     plan_feature_df['cat_max_price_mode'] = -1
#     plan_feature_df['cat_min_price_mode'] = -1
#     plan_feature_df['cat_max_eta_mode'] = -1
#     plan_feature_df['cat_min_eta_mode'] = -1
#
#     return plan_feature_df



# def gen_plan_feas(data):
#     """
#     计划特征 [max min mean std] * 3 + 8 mode
#     :param data:
#     :return:
#     """
#     # plans_df = get_plan_df(data)
#     # tr_plans + te_plans =583625
#     plans_df = pd.read_csv('../../input/kdd2019_regular/phase1/plans_djsd.csv')
#     plans_features = gen_plan_feas_by_plans(plans_df)
#
#     data_empty = data[~data['sid'].isin(plans_df.sid.unique())]
#     empty_plans_features = gen_empty_plan_feas(data_empty)
#     plan_feature_df = pd.concat([plans_features, empty_plans_features], axis=0).reset_index(drop=True)
#     print('mode tfidf...')
#     tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
#     # 添加tdidf svd
#     print 'ljj here checking,', plan_feature_df['mode_texts']
#     tfidf_vec = tfidf_enc.fit_transform(plan_feature_df['mode_texts'])
#
#     svd_enc = TruncatedSVD(n_components=30, n_iter=50, random_state=2019)
#     mode_svd = svd_enc.fit_transform(tfidf_vec)
#     mode_svd_df = pd.DataFrame(mode_svd)
#     mode_svd_df.columns = ['svd_mode_{}'.format(i) for i in range(30)]
#     feature_df = pd.concat([plan_feature_df, mode_svd_df], axis=1)
#     return feature_df

def gen_user_feas(data):

    data = data.sort_values(['cat_pid','req_time'])
    print data.head(),data.shape

    def gen_pre_list(x,column):
        le = x.size
        pre_list = list(map(list, zip(*[x.shift(i).values for i in range(1, 1 + le)][::-1])))
        ls = {column: pd.Series(pre_list)}
        df = pd.DataFrame(ls, columns=[column])
        df.index = x.index
        return df

    data['gen_pre_mode_list'] = data.groupby('cat_pid')['click_mode'].apply(gen_pre_list,'gen_pre_mode_list')
    data['gen_pre_req_time_list'] = data.groupby('cat_pid')['req_time'].apply(gen_pre_list,'gen_pre_req_time_list')
    print data.head()


    data['cat_last_click_mode'] = data.gen_pre_mode_list.apply(lambda x: x[-1])
    data['cat_last_click_mode'] = data.cat_last_click_mode.fillna(0)

    data['num_last_req_time'] = data.gen_pre_req_time_list.apply(lambda x:x[-1])
    data['num_last_req_time'] = data['num_last_req_time'].fillna(data.req_time.min())
    data['num_how_long_till_this_time'] = data['req_time'].astype(int) - data['num_last_req_time'].astype(int)

    def mode_num(x):
        """
        :param c:
        :return:
        """
        if x is np.nan:
            return [0] * 12
        if -1 in x:
            x = x[0:x.index(-1)]
        c = pd.value_counts(x)
        z = np.zeros(12)
        if len(c) > 0:
            k = c.index.values.astype(np.int32)
            v = c.values
            z[k] = v
            z = z / np.sum(z)
        return z

    mode_num_names = ['cat_mode_num_{}'.format(i) for i in range(12)]
    pid_group_df = data['gen_pre_mode_list'].apply(lambda x: mode_num(x)).reset_index()
    mode_columns = ['sid'] + mode_num_names
    mode_data = np.concatenate(pid_group_df['gen_pre_mode_list'].values, axis=0).reshape(len(pid_group_df), 12)
    sid_data = data['sid'].values.reshape(len(data), 1)
    mode_num_df = pd.DataFrame(np.hstack([sid_data, mode_data]), columns=mode_columns)
    mode_num_df.columns = mode_columns
    data = pd.merge(data, mode_num_df, on=['sid'], how='left')

    def get_max_fre(x):
        if x is np.nan:
            return np.nan
        if -1 in x:
            x = x[0:x.index(-1)]
        c = pd.value_counts(x)
        if len(c) == 0:
            return np.nan
        else:
            return c.idxmax()

    data['cat_pid_max_mode'] = data['gen_pre_mode_list'].apply(get_max_fre)
    return data

def gen_time_feas(data):
    data['num_time_diff'] = data['plan_time'].astype(int) - data['req_time'].astype(int)
    data['cat_weekday'] = data['req_time'].dt.dayofweek
    data['cat_is_weekday'] = data['req_time'].dt.dayofweek.apply(lambda x: 0 if x in [1, 2, 3, 4, 5] else 1)
    data['cat_month'] = data['req_time'].dt.month
    data['cat_dayofyear'] = data['req_time'].dt.dayofyear
    data['cat_timeofday'] = data['req_time'].dt.hour.apply(
        lambda x: 0 if x <= 6 else 1 if x <= 12 else 2 if x <= 18 else 3)
    # # data['cat_holiday'] =
    data['cat_hour'] = data['req_time'].dt.hour
    data = data.drop(['req_time'], axis=1)
    data = data.drop(['plan_time'], axis=1)

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

    te_click = pd.read_csv('preds/20190506-1648-lgb-tst1-0.67593-test-fulltrain.csv') # 迁移学习一下
    print te_click.columns

    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)
    te_data = te_queries.merge(te_click,on='sid',how='left')
    te_data = te_data.merge(te_plans, on='sid', how='left')
    # te_data['click_mode'] = -1

    tr_data['origin_click_mode'] = tr_data['click_mode']
    te_data['origin_click_mode'] = -1
    data = pd.concat([tr_data, te_data], axis=0)
    # data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)

    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def split_train_val(data):
    modified_array = np.delete(data.columns.values, np.where(data.columns.values == 'click_mode'))
    X = data[list(modified_array)].values
    y = data[['click_mode']].values
    from sklearn.model_selection import train_test_split

    print X.shape
    print y.shape
    print data.shape
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.22, random_state=42)

    print X_train.shape
    print  y_train.shape
    print y_train.T.shape
    train = pd.DataFrame(np.concatenate((y_train, X_train), axis=1), columns=data.columns)
    val = pd.DataFrame(np.concatenate((y_val, X_val), axis=1), columns=data.columns)
    return train, val

# def gen_plan_df(data):
#     """
#     [1, 2, 7, 9, 11] price 存在''
#     [3,5,6] price全是0
#     4 8 10 price不为''
#     对数据中的plans展开，生成plans dataframe
#     对plans 进行预处理，填充
#     :return:
#     """
#     # data.loc[data['click_mode'] != -1, 'is_train'] = 1
#     # data.loc[data['click_mode'] == -1, 'is_train'] = 0
#     data.loc[data['plans'].isnull(), 'plans'] = '[]'
#     data['plans'] = data['plans'].apply(lambda line: eval(line))
#     lens = [len(item) for item in data['plans']]
#     # plan_time distance eta price transport_mode
#     sid_list = np.repeat(data['sid'].values, lens)
#     plan_time_list = np.repeat(data['plan_time'].values, lens)
#     plans = np.concatenate(data['plans'].values)
#     plan_pos = np.concatenate([list(range(1, l + 1)) for l in lens])
#     df_data = []
#     for s, t, p in zip(sid_list, plan_time_list, plans):
#         p['sid'] = s
#         p['plan_time'] = t
#         df_data.append(p)
#     # 生成新的plans_df
#     plans_df = pd.DataFrame(df_data)
#     plans_df = plans_df[['sid', 'plan_time', 'distance', 'eta', 'price', 'transport_mode']]
#     plans_df['plan_time'] = pd.to_datetime(plans_df['plan_time'])
#     # '' 替换成np.nan
#     plans_df['price'] = plans_df['price'].replace(r'', np.NaN)
#     plans_df['plan_pos'] = plan_pos
#
#     ###############
#     def convert_time(d, m):
#         return (d.hour * 60 + d.minute) // m
#
#     # 3 5 6 价格填充为0
#     plans_df.loc[plans_df['transport_mode'].isin([3, 5, 6]), 'price'] = 0
#     plans_df['time_num30'] = plans_df['plan_time'].apply(lambda x: convert_time(x, 30))
#     # 计算单价和mode平均单价
#     plans_df['dj'] = plans_df['price'] / plans_df['distance']
#     # 单价>3大多是因为距离太近导致，只有60个；只有13条异常数据
#     # mode 3 有两条异常数据，估计使收取过路费
#     plans_df.loc[(plans_df['transport_mode'] == 4) & (plans_df['dj'] > 3), 'dj'] = 3
#     plans_df['mdj'] = plans_df.groupby(['transport_mode', 'time_num30'])['dj'].transform(lambda x: np.nanmedian(x))
#     # 填充 price dj[1, 2, 7, 9, 11]
#     # 用平均单价替换价格
#     plans_df.loc[plans_df['price'].isnull(), 'dj'] = plans_df.loc[plans_df['price'].isnull(), 'mdj']
#     df2 = plans_df.loc[plans_df['price'].isnull()]
#     # 价格为''的 用单价*距离代替价格
#     plans_df.loc[plans_df['price'].isnull(), 'price'] = df2['dj'] * df2['distance']
#     # 生成 速度和 速度/单价比
#     plans_df['sd'] = plans_df['distance'] / plans_df['eta']
#     plans_df['sd_dj'] = plans_df['sd'] / plans_df['dj']
#     # sid, plan_time, distance, eta, price, transport_mode ；最高性价比
#     plans_df.loc[plans_df['sd_dj'] > 1000, 'sd_dj'] = 1000
#     return plans_df[['sid', 'plan_time', 'plan_pos', 'distance', 'eta', 'price', 'transport_mode', 'dj', 'sd', 'sd_dj']]



if True:
    print 'merge_raw_Dataing '
    data = merge_raw_data()
    data.rename(columns={'pid':'cat_pid'},inplace=True)
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['plan_time'] = pd.to_datetime(data['plan_time'])

    print 'merge user feas'
    data = gen_user_feas(data)
    data.to_csv('../../input/kdd2019_regular/phase1/data_user.csv',index=False)

    print 'gen_od_feas '

    data = gen_od_feas(data)
    print 'gen_profile_feas '
    data = gen_profile_feas(data)

    print 'gen_plan_feas '
    # plans_features = gen_plan_feas(data)
    # union没有plans的 innner=left
    # data = pd.merge(data, plans_features, on=['sid'], how='left')
    data = gen_plan_feas_origin(data)

    print 'gen_time_feas '

    data = gen_time_feas(data)

    data.to_csv('../../input/kdd2019_regular/phase1/data.csv')
    # data=pd.read_csv('../../input/kdd2019_regular/phase1/data.csv', index_col=0)


    data[data.origin_click_mode != -1].to_csv('../../input/kdd2019_regular/phase1/train.csv')

    train, val = split_train_val(data[data.origin_click_mode != -1])
    train.to_csv('../../input/kdd2019_regular/phase1/train1.csv')
    val.to_csv('../../input/kdd2019_regular/phase1/val1.csv')

    data[data.origin_click_mode == -1].to_csv('../../input/kdd2019_regular/phase1/test.csv')

    for name in ['train', 'test', 'train1', 'val1']:
        print("Processing %s..." % name)
        data = pd.read_csv('../../input/kdd2019_regular/phase1/%s.csv' % name, index_col=0)

        # Save column names
        if name in ['train']:
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
