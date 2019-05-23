# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
pd.options.display.max_rows=999
pd.options.display.max_columns = 999

from utils import Dataset
from sklearn.decomposition import TruncatedSVD

def gen_user_feas(data):

    data = data.sort_values(['pid','req_time']) # already sorted
    print data.head(),data.shape

    def gen_pre_list(x,column):
        le = x.size
        pre_list = list(map(list, zip(*[x.shift(i).values for i in range(1, 1 + le)][::-1])))
        ls = {column: pd.Series(pre_list)}
        df = pd.DataFrame(ls, columns=[column])
        df.index = x.index
        return df

    data['gen_pre_mode_list'] = data.groupby('pid')['click_mode'].apply(gen_pre_list,'gen_pre_mode_list')
    print data.head()
    data['gen_pre_req_time_list'] = data.groupby('pid')['req_time'].apply(gen_pre_list,'gen_pre_req_time_list')
    print data.head()

    data['cat_last_click_mode'] = data.gen_pre_mode_list.apply(lambda x: x[-1])
    data['cat_last_click_mode'] = data.cat_last_click_mode.fillna(0)

    data['last_req_time'] = data.gen_pre_req_time_list.apply(lambda x:x[-1])
    data['last_req_time'] = data['last_req_time'].fillna(data.req_time.min())

    data['cat_last_request_weekday'] = data['last_req_time'].dt.dayofweek
    data['cat_last_request_hour'] = data['last_req_time'].dt.hour

    data['num_how_long_till_this_time'] = data['req_time'].astype(int) - data['last_req_time'].astype(int)

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

    def get_max_mode_pre(x):
        if x is np.nan:
            return np.nan
        if -1 in x:
            x = x[0:x.index(-1)]
        c = pd.value_counts(x)
        if len(c) == 0:
            return np.nan
        else:
            return c.idxmax()

    data['cat_pid_max_mode'] = data['gen_pre_mode_list'].apply(get_max_mode_pre)
    data.cat_pid_max_mode =data.cat_pid_max_mode.fillna(0)
    return data




print("Loading data...")

data = Dataset.load_part('data','manual')
feature = Dataset.get_part_features('manual_data')

data_df = pd.DataFrame(data,columns=feature)

data_df['req_time'] = pd.to_datetime(data_df['req_time'])
data_df['plan_time'] = pd.to_datetime(data_df['plan_time'])

# data_df = data_df.loc[:1000]

result = gen_user_feas(data_df)

cat_columns = [c for c in result.columns if c.startswith('cat')]
num_columns = [c for c in result.columns if c.startswith('num')]
print('cat_columns',cat_columns)
print('num_columns',num_columns)

Dataset.save_part_features('categorical_pid', cat_columns)
Dataset.save_part_features('numeric_time', num_columns)

Dataset(categorical=result[cat_columns].values).save('data_pid')
Dataset(numeric=result[num_columns].values).save('data_pid')

print('Done!')
