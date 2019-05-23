# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import Dataset
from geopy.distance import geodesic


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

print("Loading data...")

data = Dataset.load_part('data','manual')
feature = Dataset.get_part_features('manual_data')

data_df = pd.DataFrame(data,columns=feature)

result = gen_od_feas(data_df)

cat_columns = [c for c in result.columns if c.startswith('cat')]
num_columns = [c for c in result.columns if c.startswith('num')]
print('cat_columns',cat_columns)
print('num_columns',num_columns)

Dataset.save_part_features('categorical_od', cat_columns)
Dataset.save_part_features('numeric_od', num_columns)

Dataset(categorical=result[cat_columns].values).save('od')
Dataset(numeric=result[num_columns].values).save('od')

print('Done!')
