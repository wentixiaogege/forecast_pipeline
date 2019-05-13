# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

seefull = pd.read_csv('preds/20190510-1745-lgb-tst1-0.68019-test_foldavg_p_backup.csv')
seefull.rename(columns={'Unnamed: 0':'sid'},inplace=True)

seefull['click_mode'] = seefull.apply(lambda x:np.argmax(list(x[1:])),axis=1)

check=0
for i in range(1,11):
    print 'processing ',i
    aaa = seefull[(seefull.click_mode ==i)&(seefull[str(i)] < 0.4)]
    check = check + aaa.shape[0]
    print aaa.shape
    seefull.loc[(seefull.click_mode ==i)&(seefull[str(i)] < 0.4),'click_mode'] = 0

print check

seefull[['sid','click_mode']].to_csv('preds/post_process20190510-1745-lgb-tst1-0.68019-test_foldavg_p_backup.csv',index=False, header=True)
