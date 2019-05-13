# -*- coding: utf-8 -*-

import pandas as pd

from utils import Dataset
from geopy.distance import geodesic

print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m) #计算两个坐标直线距离
print(geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).km) #计算两个坐标直线距离


locations = pd.read_csv('../../input/kdd2019_regular/phase1/lntlat_adress_6525.csv')
location_columns =['adcode','district','lntlat','street','street_number']
locations=locations[location_columns]

used_features=['num_o1','num_o2','num_d1','num_d2']
#计算
for name in ['train', 'test']:
    print("Processing %s..." % name)

    num = pd.DataFrame(Dataset.load_part(name, 'numeric'), columns=Dataset.get_part_features('numeric'))
    num = num[used_features]
    num['source_lntlat'] = num.num_o1.apply(lambda x:'%.2f' % x)+',' + num.num_o2.apply(lambda x:'%.2f' % x)
    num['des_lntlat'] = num.num_d1.apply(lambda x: '%.2f' % x) + ',' + num.num_d2.apply(lambda x: '%.2f' % x)

    locations.columns  = 'source_'+location_columns
    merge = pd.merge(num,locations,left_on=['source_lntlat'],right_on=['lntlat'],how='inner')

    locations.columns  = 'des_'+location_columns

    merge = pd.merge(merge,locations,left_on=['des_lntlat'],right_on=['lntlat'],how='inner')

    merge['cat_same_district'] = merge.apply(lambda x:1 if x['source_district'] == x['des_district'] else 0)
    merge['cat_same_adcode'] = merge.apply(lambda x:1 if x['source_adcode'] == x['des_adcode'] else 0)
    merge['num_direct_distance'] = merge.apply(lambda x:geodesic((x.num_o2,x.num_o1),(x.num_d2,x.num_d2)).m)


    print name,'Done..'
    Dataset(cat_query_features=merge.values).save(name)
    if name == 'train':
        Dataset.save_part_features('cat_query_features', list(merge.columns))

print("Done.")
