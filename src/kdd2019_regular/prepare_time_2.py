# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import Dataset
from sklearn.decomposition import TruncatedSVD

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



print("Loading data...")

data = Dataset.load_part('data','manual')
feature = Dataset.get_part_features('manual_data')

data_df = pd.DataFrame(data,columns=feature)

data_df['req_time'] = pd.to_datetime(data_df['req_time'])
data_df['plan_time'] = pd.to_datetime(data_df['plan_time'])

result = gen_time_feas(data_df)

cat_columns = [c for c in result.columns if c.startswith('cat')]
num_columns = [c for c in result.columns if c.startswith('num')]
print('cat_columns',cat_columns)
print('num_columns',num_columns)

Dataset.save_part_features('categorical_time', cat_columns)
Dataset.save_part_features('numeric_time', num_columns)

Dataset(categorical=result[cat_columns].values).save('time')
Dataset(numeric=result[num_columns].values).save('time')

print('Done!')
