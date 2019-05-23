# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
pd.options.display.max_rows=999
pd.options.display.max_columns = 999

from utils import Dataset,hstack

if False:
    print("Loading data...")

    split_list=[('data','manual'),('profile','categorical'),('profile','svd'),
                ('time','categorical'),('time','numeric'),('od','categorical'),
                ('od','numeric'),('plan','categorical'),('plan','numeric'),
                ('plan','svd')]

    feature_parts = [Dataset.load_part(ds, part) for ds,part in split_list]


    feature_names = [part+'_'+ds for ds,part in split_list]
    column_names=[]
    for name in feature_names:
        column_names += Dataset.get_part_features(name)

    print feature_names

    data_df = pd.DataFrame(hstack(feature_parts),columns=column_names)

    print data_df.head()

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


    data_df[data_df.origin_click_mode != -1].to_csv('../../input/kdd2019_regular/phase1/train.csv')

    train, val = split_train_val(data_df[data_df.origin_click_mode != -1])
    train.to_csv('../../input/kdd2019_regular/phase1/train1.csv')
    val.to_csv('../../input/kdd2019_regular/phase1/val1.csv')

    data_df[data_df.origin_click_mode == -1].to_csv('../../input/kdd2019_regular/phase1/test.csv')

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
    Dataset(id=data['sid']).save(name)

    Dataset(svd=data[svd_columns].values).save(name)

    if 'click_mode' in data.columns:
        Dataset(target=data['click_mode']).save(name)

print("Done.")