# -*- coding: utf-8 -*-
__author__ = 'lijingjie'

import sys
sys.path.insert(0, 'src/models/')
sys.path.insert(0, 'src/conf/')
sys.path.insert(0, '../conf/')
sys.path.insert(0, '../models')
sys.path.insert(0, '../')
import graphviz
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['LIGHTGBM_EXEC'] = "/Users/jacklee/LightGBM/lightgbm"
# os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin'

import lightgbm as lgb

bst = lgb.Booster(model_file='lightgbm/20190512-1047/lgb-lgb-tst1-fold-0-0.dump')

image = lgb.create_tree_digraph(bst, tree_index=1,show_info=['split_gain','internal_value','internal_count','leaf_count'])

image.render('lightgbm/20190512-1047/lgb-lgb-tst1-fold-0-0.gv', view=True)

print ('checking Done!')
