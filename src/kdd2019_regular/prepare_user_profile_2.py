# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from utils import Dataset
from sklearn.decomposition import TruncatedSVD



def read_profile_data():
    profile_data = pd.read_csv('../../input/kdd2019_regular/phase1/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1  # 增加pid=-1的
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    profile_data.columns = ['cat_' + i if i!='pid' else 'pid' for i in profile_data.columns]
    return profile_data


def gen_profile_feas(data):
    profile_data = read_profile_data()
    print profile_data.head()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=10, n_iter=30, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_profile_fea_{}'.format(i) for i in range(10)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1) # nan的pid 搞成了-1
    data = data.merge(svd_feas, on='pid', how='left')
    limit_profile_data = profile_data[['pid','cat_p13','cat_p29','cat_p33','cat_p9','cat_p6','cat_p5','cat_p0']] # 这些feature对0类别应该会有好的效果
    data = data.merge(limit_profile_data, on='pid', how='left') # ---> adding origin pid features
    return data


print("Loading data...")

data = Dataset.load_part('data','manual')
feature = Dataset.get_part_features('manual_data')

data_df = pd.DataFrame(data,columns=feature)

result = gen_profile_feas(data_df)
result.rename(columns={'pid':'cat_pid'},inplace=True)

cat_columns = [c for c in result.columns if c.startswith('cat')]
svd_columns = [c for c in result.columns if c.startswith('svd')]
print('cat_columns',cat_columns)
print('svd_columns',svd_columns)

Dataset.save_part_features('categorical_profile', cat_columns)
Dataset.save_part_features('svd_profile', svd_columns)

Dataset(categorical=result[cat_columns].values).save('profile')
Dataset(svd=result[svd_columns].values).save('profile')

print('Done!')
