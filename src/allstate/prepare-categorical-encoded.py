# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from tqdm import tqdm
from utils import Dataset

print("Loading data...")

train_cat = Dataset.load_part('train', 'categorical')
test_cat = Dataset.load_part('test', 'categorical')

train_cat_enc = np.zeros(train_cat.shape, dtype=np.uint8) #多维度的
test_cat_enc = np.zeros(test_cat.shape, dtype=np.uint8)

# 对所有的categories 做了计数encoding
with tqdm(total=train_cat.shape[1], desc='  Encoding', unit='cols') as pbar:
    for col in range(train_cat.shape[1]):
        values = np.hstack((train_cat[:, col], test_cat[:, col]))
        values = np.unique(values)
        values = sorted(values, key=lambda x: (len(x), x))

        encoding = dict(list(zip(values, list(range(len(values))))))#类别编号在此

        train_cat_enc[:, col] = pd.Series(train_cat[:, col]).map(encoding).values
        test_cat_enc[:, col] = pd.Series(test_cat[:, col]).map(encoding).values

        pbar.update(1)

print("Saving...")

Dataset.save_part_features('categorical_encoded', Dataset.get_part_features('categorical'))
print(train_cat_enc)
Dataset(categorical_encoded=train_cat_enc).save('train')
Dataset(categorical_encoded=test_cat_enc).save('test')

print("Done.")
