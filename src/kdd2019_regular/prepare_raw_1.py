# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils import Dataset

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


if True:
    print 'merge_raw_Dataing '
    data = merge_raw_data()
    # data.rename(columns={'pid':'cat_pid'},inplace=True)
    data['pid'] = data['pid'].fillna(-1) # nan的pid 搞成了-1
    data = data.sort_values(['pid','req_time'])

    Dataset(manual=data[data.columns].values).save('data')
    Dataset.save_part_features('manual_data', list(data.columns))

    print("Done.")
