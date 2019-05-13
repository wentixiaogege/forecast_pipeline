# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import Dataset

print("Loading data...")

train_cat = Dataset.load_part('train', 'categorical')
test_cat = Dataset.load_part('test', 'categorical')

train_cat_counts = np.zeros(train_cat.shape, dtype=np.float32)
test_cat_counts = np.zeros(test_cat.shape, dtype=np.float32)

#1.将训练和预测中类别计数，保存
#
with tqdm(total=train_cat.shape[1], desc='  Counting', unit='cols') as pbar:
    for col in range(train_cat.shape[1]):
        train_series = pd.Series(train_cat[:, col])
        test_series = pd.Series(test_cat[:, col])

        counts = pd.concat((train_series, test_series)).value_counts()
        train_cat_counts[:, col] = train_series.map(counts).values
        test_cat_counts[:, col] = test_series.map(counts).values


        pbar.update(1)

print("Saving...")

Dataset.save_part_features('categorical_counts', Dataset.get_part_features('categorical'))
Dataset(categorical_counts=train_cat_counts).save('train')
Dataset(categorical_counts=test_cat_counts).save('test')

print("Done.")

print("Loading data...")

train_cat = Dataset.load_part('train1', 'categorical')
test_cat = Dataset.load_part('val1', 'categorical')

train_cat_counts = np.zeros(train_cat.shape, dtype=np.float32)
test_cat_counts = np.zeros(test_cat.shape, dtype=np.float32)

#1.将训练和预测中类别计数，保存
#
with tqdm(total=train_cat.shape[1], desc='  Counting', unit='cols') as pbar:
    for col in range(train_cat.shape[1]):
        train_series = pd.Series(train_cat[:, col])
        test_series = pd.Series(test_cat[:, col])

        counts = pd.concat((train_series, test_series)).value_counts()
        train_cat_counts[:, col] = train_series.map(counts).values
        test_cat_counts[:, col] = test_series.map(counts).values


        pbar.update(1)

print("Saving...")

Dataset.save_part_features('categorical_counts', Dataset.get_part_features('categorical'))
Dataset(categorical_counts=train_cat_counts).save('train1')
Dataset(categorical_counts=test_cat_counts).save('val1')

print("Done.")
