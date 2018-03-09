import os
import sys
import random
import math
import re
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import copy

LIB_DIR = os.path.expanduser('~/Dropbox/lib/')
sys.path.append(LIB_DIR)
MRCNN_DIR = os.path.join(LIB_DIR, "Mask_RCNN")
sys.path.append(MRCNN_DIR)

from Mask_RCNN.config import Config
import Mask_RCNN.utils as mrnn_utils
import Mask_RCNN.model as mrnn_modellib
import Mask_RCNN.visualize as visualize
from Mask_RCNN.model import log

sys.path.append("../")
from src.config import NucleiConfig
from src.utils import *
from src.dataset import NucleiDataset

import warnings; warnings.simplefilter('ignore')

ROOT_DIR = ("../")
DATASET_DIR = os.path.join(ROOT_DIR, 'data')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MRCNN_DIR, "mask_rcnn_coco.h5")


def train_nuclei(model, train_ids, val_ids, epoch_list=[], layer_list=[], lr_list=[]):
    # Training dataset
    dataset_train = NucleiDataset()
    dataset_train.load_dataset(DATASET_DIR, 'stage1_train', image_ids = train_ids)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleiDataset()
    dataset_val.load_dataset(DATASET_DIR, 'stage1_train', image_ids = val_ids)
    dataset_val.prepare()
    for epoch, layer, lr in zip(epoch_list, layer_list, lr_list):
        print(format("Training layers: %s at lr = %s until epoch %d" % (layer, lr, epoch)))
        model.train(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=epoch,
                    layers=layer)
    return model


def detect_nuclei(model, dataset, no_overlap=True):
    """Run detection model on datasets and return the results.
    """
    results = dict()
    for image_id in dataset.image_ids:
        # Load image and ground truth data
        image = dataset.load_image(image_id)
        # Run object detection
        r = model.detect([image], verbose=0)[0]
        results[image_id] = r
        if no_overlap:
            if not len(r['scores']) == 0:
                r['masks'], overlaps = remove_overlap(r['masks'], r['scores'])
    return results
