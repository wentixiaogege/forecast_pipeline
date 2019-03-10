# -*- coding: utf-8 -*-
import numpy as np

from scipy.stats import skew, boxcox

from tqdm import tqdm
from utils import Dataset

print("Loading data...")

train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

train_num_enc = np.zeros(train_num.shape, dtype=np.float32)
test_num_enc = np.zeros(test_num.shape, dtype=np.float32)

with tqdm(total=train_num.shape[1], desc='  Transforming', unit='cols') as pbar:
    for col in range(train_num.shape[1]):
        values = np.hstack((train_num[:, col], test_num[:, col]))

        sk = skew(values)  #https://blog.csdn.net/xbmatrix/article/details/69360167

        if sk > 0.25:#如果skew很大，就做boxcox变幻
            #Box-Cox变换是Box和Cox在1964年提出的一种广义幂变换方法，是统计建模中常用的一种数据变换，用于连续的响应变量不满足正态分布的情况。Box-Cox变换之后，可以一定程度上减小不可观测的误差和预测变量的相关性。Box-Cox变换的主要特点是引入一个参数，通过数据本身估计该参数进而确定应采取的数据变换形式，Box-Cox变换可以明显地改善数据的正态性、对称性和方差相等性，对许多实际数据都是行之有效的
            values_enc, lam = boxcox(values+1)

            train_num_enc[:, col] = values_enc[:train_num.shape[0]]
            test_num_enc[:, col] = values_enc[train_num.shape[0]:]
        else:
            train_num_enc[:, col] = train_num[:, col]
            test_num_enc[:, col] = test_num[:, col]

        pbar.update(1)

print("Saving...")

Dataset.save_part_features('numeric_boxcox', Dataset.get_part_features('numeric'))
Dataset(numeric_boxcox=train_num_enc).save('train')
Dataset(numeric_boxcox=test_num_enc).save('test')

print("Done.")
