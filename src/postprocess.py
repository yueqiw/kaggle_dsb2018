import json
import math
import re
import time
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os, sys
import numpy as np
import pandas as pd
from imagecluster import main
from imagecluster.imagecluster import cluster


def merge_multiple_rles(rle_dict, data_ids, dataset='test', parent_dir='../detections'):
    '''merge rles from different models
    rle_dict:  {"xxx/xxx.csv": [vvvvv, yyyyy, zzzzz, ...], ...}
    '''
    id_use = []
    rle_df_list = []
    model_list = []
    for fname, id_list in rle_dict.items():
        id_list = list(id_list)
        pred_name = os.path.split(fname)[1].strip(".csv")
        print(fname)
        model_name = re.findall(r'logs/(nuclei.*)/', os.path.split(fname)[0])[0]
        rle = pd.read_csv(fname)
        rle_use = rle.loc[np.isin(rle['ImageId'], id_list), ].copy()
        rle_use['source'] = pred_name
        id_use += id_list
        model_list.append(model_name)
        rle_df_list.append(rle_use)
    rle_all = pd.concat(rle_df_list)
    assert(len(id_use) == len(set(id_use)))
    assert(set(id_use) == set(data_ids))
    dir_name = dataset + '_' + '_'.join(model_list)
    dir_path = os.path.join(parent_dir, dataset + '_' + '_'.join(model_list))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    rle_all[['ImageId', 'EncodedPixels']].to_csv(os.path.join(dir_path, dir_name + '_submit.csv'), index=False)
    rle_all.to_csv(os.path.join(dir_path, dir_name + '.csv'), index=False)
    return rle_all


def mask_voting(masks1, masks2, scores1, scores2):
    ious = mask_overlaps_rle2(masks1, masks2)
    max_iou = np.max(ious, axis=1)
    max_idx = np.argmax(ious, axis=1)
    matched = max_iou > 0.7
    ids1 = np.where(matched)[0]
    ids1_not_matched = [x for x in range(len(scores1)) if not x in ids1]
    ids2 = max_idx[matched]
    ids2_not_matched = [x for x in range(len(scores2)) if not x in ids2]
    sc1 = scores1[matched]
    sc2 = scores2[matched]
    overlap_masks = []
    for i, x1, x2 in enumerate(zip(ids1, ids2)):
        if scores1[x1] > scores2[x2]:
            good_mask = masks1[x1]
        else:
            good_mask = masks2[x2]
        overlap_masks.append(good_mask)
    return overlap_masks, masks1[ids1_not_matched], masks2[ids2_not_matched]
