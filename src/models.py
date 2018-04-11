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
import pickle
import json
import datetime
import shutil
import seaborn as sns

LIB_DIR = os.path.expanduser('~/Dropbox/lib/')
sys.path.append(LIB_DIR)

from Mask_RCNN.config import Config
import Mask_RCNN.utils as mrnn_utils
import Mask_RCNN.model as mrnn_modellib
import Mask_RCNN.visualize as visualize
from Mask_RCNN.model import log

sys.path.append("../")
from src.config import NucleiConfig
from src.utils import *
from src.dataset import NucleiDataset

from keras import backend as K
import warnings; warnings.simplefilter('ignore')

ROOT_DIR = ("../")
DATASET_DIR = os.path.join(ROOT_DIR, 'data')
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
R101_COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_coco.h5")
R50_IMAGENET_MODEL_PATH = os.path.join(ROOT_DIR, "models", "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

class NucleiModel():
    def build_model(self, mode, config, architecture):
        # Create model in training mode
        return mrnn_modellib.MaskRCNN(mode=mode, config=config,
                                    architecture=architecture,
                                    model_dir=self.model_root_dir)

    def update_config(self, current_config, config_dict):
        config = copy.copy(current_config)
        for k, v in config_dict.items():
            config.__setattr__(k, v)
        if self.mode == "inference":
            config.__setattr__("GPU_COUNT", 1)
            config.__setattr__("IMAGES_PER_GPU", 1)
        config.__init__()
        return config

    def save_config(self, config, filepath):
        """Write object to pickle. can be reloaded."""
        with open(filepath, 'wb') as f:
            pickle.dump(config, f, 2)
        #print(format("Config written to pickle file %s" % filepath))
        txtpath = filepath.replace(".pkl", ".txt")
        jsonpath = filepath.replace(".pkl", ".json")
        assert txtpath.endswith(".txt") and jsonpath.endswith(".json")
        config.write_text(txtpath)
        config.write_json(jsonpath)

    def load_train_config(self):
        assert os.path.exists(self.train_config_path), "Train config file not found."
        with open(self.train_config_path, 'rb') as f:
            config = pickle.load(f)
        return config

    def save_dataset(self, dataset, filename):
        with open(filename, 'w') as f:
            json.dump(dataset, f)

    def load_dataset(self, filename):
        with open(filename, 'r') as f:
            dataset = json.load(f)
        return dataset


class NucleiModelTrain(NucleiModel):
    def __init__(self, init_with, train_ids=None, val_ids=None, dataset_id='stage1_train',
                 augment_dataset=None, augment_probability=None,
                 content_clusters=None, style_clusters=None,
                 config_dict=None, model_name=None, checkpoint=None,
                 model_dir=MODEL_DIR):
        self.mode = "train"
        self.model_root_dir = model_dir
        self.init_with = init_with
        self.model_name = model_name
        self.augment_dataset = augment_dataset
        self.augment_probability = augment_probability
        self.content_clusters = content_clusters
        self.style_clusters = style_clusters


        assert init_with in ["nuclei-pretrained", "r101-coco", "last", "checkpoint"]
        if init_with == "nuclei-pretrained" or init_with == "r101-coco":
            self.architecture = "resnet101"
            ori_config = NucleiConfig()
            self.train_config = self.update_config(ori_config, config_dict)
            self.train_config.__setattr__("architecture", self.architecture)
            self.train_config.__setattr__("init_with", init_with)

            self.model = self.build_model(mode="training", config=self.train_config,
                                        architecture=self.architecture)
            self.log_dir = self.model.log_dir
            init_checkpoint_path = self.model.checkpoint_path
            self.load_weights(model_name=model_name, checkpoint=checkpoint)
            self.model.epoch = 0
            self.model.log_dir = self.log_dir  # in case of pretrained, avoid setting it to old log_dir
            self.model.checkpoint_path = init_checkpoint_path  # # in case of pretrained, avoid setting it to old path
            self.model_name = self.log_dir.split("/")[-1]
            #self.train_config.__setattr__("log_dir", self.log_dir)
            #self.train_config.__setattr__("train_model_name", self.model_name)
            self.train_config_path = os.path.join(self.log_dir, "train_config.pkl")

            self.set_dataset(train_ids, val_ids, dataset_id)
            self.dataset_path = os.path.join(self.log_dir, "train_dataset.json")

        elif init_with == "last" or init_with == "checkpoint":
            self.log_dir = os.path.join(self.model_root_dir, model_name)
            self.train_config_path = os.path.join(self.log_dir, "train_config.pkl")
            self.train_config = self.load_train_config()
            self.architecture = self.train_config.architecture
            self.model = self.build_model(mode="training", config=self.train_config,
                                            architecture=self.architecture)
            self.load_weights(model_name=model_name, checkpoint=checkpoint)
            assert self.log_dir == self.model.log_dir  # only after loading weights

            self.dataset_path = os.path.join(self.log_dir, "train_dataset.json")
            self.dataset = self.load_dataset(self.dataset_path)
            self.train_ids = self.dataset['train_ids']
            self.val_ids = self.dataset['val_ids']
            self.dataset_id = self.dataset['train_set']

        self.train_config_dict = self.train_config.to_dict()

    def set_dataset(self, train_ids, val_ids, dataset_id):
        self.dataset = dict()
        self.dataset['train_set'] = self.dataset_id = dataset_id
        self.dataset['train_ids'] = self.train_ids = train_ids
        self.dataset['val_ids'] = self.val_ids = val_ids
        return None

    def find_last_checkpoint(self, dir_name, prefix="mask_rcnn"):
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(prefix), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint



    def load_weights(self, model_name=None, checkpoint=None):
        # Which weights to start with?
        # imagenet, coco, or last or specific epoch
        if self.init_with == "r101-coco":
            # resnet 101 pretrained on coco. contains all layer weights.
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            print("loading ResNet-101 COCO model: " + R101_COCO_MODEL_PATH)
            self.model.load_weights(R101_COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif self.init_with == "last":
            # Load the last checkpoint of a given model and continue training
            model_path = self.find_last_checkpoint(os.path.join(self.model_root_dir, model_name))
            print("loading from: " + model_path)
            self.model.load_weights(model_path, by_name=True)

        elif self.init_with == "checkpoint":
            # load a specific checkpoint
            model_path = os.path.join(self.model_root_dir, model_name, checkpoint + ".h5")
            print("loading from: " + model_path)
            self.model.load_weights(model_path, by_name=True)

        elif self.init_with == "nuclei-pretrained":  # need to tranfer good models to /models
            model_path = os.path.join(ROOT_DIR, "models", model_name, checkpoint + ".h5")
            print("loading from: " + model_path)
            self.model.load_weights(model_path, by_name=True)

        return None


    def train_nuclei(self, epoch_list=[], layer_list=[], lr_list=[]):

        self.epoch_list = epoch_list
        self.layer_list = layer_list
        self.lr_list = lr_list
        # Training dataset
        self.dataset_train = NucleiDataset()
        self.dataset_train.load_dataset(DATASET_DIR, self.dataset_id, image_ids = self.train_ids)
        self.dataset_train.prepare()
        if self.augment_dataset:
            self.dataset_extra = NucleiDataset()

            print(self.augment_dataset)  # for multiple datasets, should create multiple instances instead.
            self.dataset_extra.load_dataset_style_transfer(DATASET_DIR, mask_subset = self.dataset_id,
                                transfer_subset = self.augment_dataset,
                                content_ids = self.train_ids,
                                content_clusters = self.content_clusters,
                                style_clusters = self.style_clusters)
            self.dataset_extra.prepare()
            print(self.dataset_extra.image_ids)
        else:
            self.dataset_extra = None
        print(self.dataset_train.image_ids)



        # Validation dataset
        self.dataset_val = NucleiDataset()
        self.dataset_val.load_dataset(DATASET_DIR, self.dataset_id, image_ids = self.val_ids)
        self.dataset_val.prepare()

        if self.init_with == "nuclei-pretrained" or self.init_with == "r101-coco":
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            self.save_config(self.train_config, self.train_config_path)
            self.save_dataset(self.dataset, self.dataset_path)

        self.train_config.display()

        for i, (epoch, layer, lr) in enumerate(zip(epoch_list, layer_list, lr_list)):
            if i > 0:
                last_checkpoint = self.find_last_checkpoint(self.log_dir)
                del self.model
                K.clear_session()
                self.model = self.build_model(mode="training", config=self.train_config,
                                                architecture=self.architecture)
                print("\n\nloading from: " + last_checkpoint)
                self.model.load_weights(last_checkpoint, by_name=True)
                #assert self.model.epoch == epoch - 1, "Loaded model does not match the current epoch."
            print(format("Training layers: %s at lr = %s until epoch %d" % (layer, lr, epoch)))
            self.model.train(self.dataset_train, self.dataset_val,
                        learning_rate=lr,
                        epochs=epoch,
                        layers=layer,
                        augment_dataset=self.dataset_extra,
                        augment_probability=self.augment_probability)
        return None



class NucleiModelInference(NucleiModel):
    def __init__(self, infer_ids, data_type, dataset_id, subclass_df, config_dict,
                 model_name, checkpoint, comment="", model_dir=MODEL_DIR, architecture="resnet101"):
        assert data_type in ['train', 'val', 'test']
        self.mode = "inference"
        self.model_root_dir = model_dir
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.subclass_df = subclass_df
        self.comment = comment
        self.set_dataset(infer_ids, dataset_id, data_type)

        self.log_dir = os.path.join(self.model_root_dir, model_name)
        self.train_config_path = os.path.join(self.log_dir, "train_config.pkl")
        try:
            self.train_config = self.load_train_config()
            self.inference_config = self.update_config(self.train_config, config_dict)
        except:
            ori_config = NucleiConfig()
            self.inference_config = self.update_config(ori_config, config_dict)
        try:
            self.architecture = self.train_config.architecture
        except:
            self.architecture = architecture

        self.set_inference_dir()
        self.config_path = os.path.join(self.inference_dir, "inference_config.pkl")

        self.model = self.build_model(mode="inference", config=self.inference_config,
                                        architecture=self.architecture)
        self.load_weights(model_name=model_name, checkpoint=checkpoint)
        assert self.log_dir == self.model.log_dir

        self.train_dataset_path = os.path.join(self.log_dir, "train_dataset.json")
        self.infer_dataset_path = os.path.join(self.inference_dir, "infer_dataset.json")
        self.results_path = os.path.join(self.inference_dir, "infer_result.pkl")
        self.evaluation_path = os.path.join(self.inference_dir, "infer_evaluation.pkl")
        self.evaluation_df_path = os.path.join(self.inference_dir, "infer_evaluation_df.csv")
        self.evaluation_by_class_path = os.path.join(self.inference_dir, "infer_evaluation_by_class.csv")
        self.evaluation_figure_path = os.path.join(self.inference_dir, "infer_evaluation_by_class.png")

        self.rle_path = os.path.join(self.inference_dir,
                self.model_name + '_' + self.inference_name + '.csv')
        self.rle_path_submit = os.path.join(self.inference_dir,
                self.model_name + '_' + self.inference_name + '_submit.csv')
        self.config_dict = self.inference_config.to_dict()

    def set_dataset(self, infer_ids, dataset_id, data_type):
        self.dataset = dict()
        self.infer_dataset_id = self.dataset['infer_set'] = dataset_id
        self.infer_ids = self.dataset['infer_ids'] = infer_ids
        self.data_type = self.dataset['infer_data_type'] = data_type
        self.dataset['infer_checkpoint'] = self.checkpoint
        self.dataset['model_name'] = self.model_name
        self.dataset['comment'] = self.comment
        return None

    def set_inference_dir(self):
        self.config_checksum = self.inference_config.to_md5()
        self.cksum_short = self.config_checksum[:5]

        now = datetime.datetime.now()
        self.timestamp = "{:%Y%m%dT%H%M}".format(now)[2:]
        self.inference_name = format("detect_%s_%s_%s_%s" % (self.checkpoint[-4:],
                            self.cksum_short, self.data_type, self.timestamp))
        self.inference_dir = os.path.join(self.log_dir, self.inference_name)

    def load_weights(self, model_name=None, checkpoint=None):
        # load a specific checkpoint
        model_path = os.path.join(self.model_root_dir, model_name, checkpoint + ".h5")
        try:
            print("loading from: " + model_path)
            self.model.load_weights(model_path, by_name=True)
        except:
            alternative_model_path = os.path.join(ROOT_DIR, "models", model_name, checkpoint + ".h5")
            print("try loading from: " + alternative_model_path)
            self.model.load_weights(alternative_model_path, by_name=True)
            os.makedirs(os.path.join(self.model_root_dir, model_name))
            shutil.copyfile(alternative_model_path, model_path)
        return None

    def detect_nuclei(self, no_overlap=True):
        """Run detection model on datasets and return the results.
        """

        self.dataset_infer = NucleiDataset()
        self.dataset_infer.load_dataset(DATASET_DIR, self.infer_dataset_id,
                                        image_ids = self.infer_ids)
        self.dataset_infer.prepare()

        if not os.path.exists(self.inference_dir):
            os.mkdir(self.inference_dir)
        self.save_config(self.inference_config, self.config_path)
        self.save_dataset(self.dataset, self.infer_dataset_path)

        self.results = dict()
        for image_id in self.dataset_infer.image_ids:
            # Load image and ground truth data
            image = self.dataset_infer.load_image(image_id)
            # TODO: add test time augmentation
            # Run object detection
            r = self.model.detect([image], verbose=0)[0]
            self.results[image_id] = r
            if no_overlap:
                if not len(r['scores']) == 0:
                    r['masks'], overlaps = remove_overlap(r['masks'], r['scores'])

        self.save_results(self.results_path)
        return self.results

    def save_results(self, filepath):
        """save roi's and class id's (and all proposals if returned)"""
        result_small = {key: {k:v for k, v in result.items() if not "masks" in k} \
                            for key, result in self.results.items()}
        with open(filepath, 'wb') as f:
            pickle.dump(result_small, f, 2)
        #print(format("Saved detection results to pickle file %s" % filepath))

    def test_image_in_train(self, ):
        with open(self.train_dataset_path, 'r') as f:
            self.train_dataset = json.load(f)
        self.train_ids = self.train_dataset['train_ids']
        self.val_ids = self.train_dataset['val_ids']

    def evaluate_results(self, detection_masks=True):
        if self.data_type == "test":
            ground_truth = False
            self.mask_APs = None
            self.bbox_recalls = None
        else:
            ground_truth = True
            self.mask_mAP, self.mask_APs, self.mask_prec_all = \
                    mAP_dsb2018(self.dataset_infer, self.results, self.dataset_infer.image_ids)
            #TODO compute Prec vs Recall
            self.bbox_mAP, self.bbox_APs, self.bbox_recalls = \
                    mAP_bbox(self.dataset_infer, self.results, self.dataset_infer.image_ids,
                                iou_threshold=0.75)

            infer_df = self.subclass_df[self.subclass_df.image.isin(self.infer_ids)]
            infer_df["image_id"] = infer_df.image.map(self.dataset_infer.get_int_id)
            infer_df["mask_APs"] = infer_df.image_id.map(self.mask_APs)
            infer_df["bbox_APs"] = infer_df.image_id.map(self.bbox_APs)
            infer_df["bbox_recall"] = infer_df.image_id.map(self.bbox_recalls)
            infer_df.to_csv(self.evaluation_df_path)
            mean_by_class = infer_df.groupby("subclass")[["mask_APs", "bbox_APs", "bbox_recall"]].mean()
            mean_by_class.to_csv(self.evaluation_by_class_path)

            evaluation = {
                "mask_mAP": self.mask_mAP,
                "mask_mAP_class_avg": mean_by_class.mean(0)["mask_APs"],
                "mask_APs": self.mask_APs,
                "bbox_mAP": self.bbox_mAP,
                "bbox_mAP_class_avg": mean_by_class.mean(0)["bbox_APs"],
                "bbox_APs": self.bbox_APs,
                "bbox_recalls": self.bbox_recalls,
                "bbox_recalls_class_avg": mean_by_class.mean(0)["bbox_recall"],
                "data_type": self.data_type,
                "detection_masks": detection_masks
            }
            with open(self.evaluation_path, 'wb') as f:
                pickle.dump(evaluation, f, 2)

            melted_df = pd.melt(infer_df, id_vars="subclass", var_name="metrics",
                    value_vars=["mask_APs", "bbox_APs", "bbox_recall"])
            g = sns.factorplot(x="value", y="subclass",
                       col="metrics", data=melted_df, kind="swarm", palette="Set2",
                       size=4, aspect=1)
            g.savefig(self.evaluation_figure_path)


        if detection_masks:
            generate_detection_masks(self.dataset_infer, self.results, self.inference_dir,
                    ground_truth=ground_truth, avg_precisions=self.mask_APs, recalls=self.bbox_recalls)
            move_subclass_to_folder(input_folder=self.inference_dir, copy=False,
                        output_folder=self.inference_dir, subclass_df=self.subclass_df)
        self.rle_results = rle_encode_all(self.dataset_infer, self.results, self.dataset_infer.image_ids)
        self.rle_results.to_csv(self.rle_path)
        self.rle_results[['ImageId', 'EncodedPixels']].to_csv(self.rle_path_submit, index=False)
        print(format("Saved RLE results to %s" % self.rle_path))
        return None

    #TODO save results and evaluations.
