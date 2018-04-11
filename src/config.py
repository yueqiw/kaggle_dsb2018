import os, sys, json
import pandas as pd
import pickle, hashlib
MASK_RNN_DIR = os.path.expanduser('~/Dropbox/lib/')
sys.path.append(MASK_RNN_DIR)

from Mask_RCNN.config import Config


class NucleiConfig(Config):
    NAME = 'nuclei'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    WITH_FLOAT_MASKS = False # set True for mask voting.

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    INVERT_DARK = True

    # Use smaller anchors because our image and objects are small
    # This is in the scale of raw pixels.
    # should adjust based on image dimension and object size
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # effective filter size
    # BACKBONE_STRIDES * (rpn_filter_size + 1) - 1 = BACKBONE_STRIDES * 4 - 1
    # (15, 31, 63, 127, 255)

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    TRAIN_ROIS_PER_IMAGE = 512

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    STEPS_PER_EPOCH = 200

    VALIDATION_STEPS = 25

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400
    POST_NMS_ROIS_INFERENCE = 2000

    # For diagnosis only. To extract all ROI proposals rather than the positive ones.
    DETECTION_EXTRACT_ALL_PROPOSALS = True

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    def to_string(self):
        """Write Configuration values to a string"""
        result = ""
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                result += "{:30} {}\n".format(a, getattr(self, a))
        return result

    def write_text(self, filename):
        """Write Configuration values. for display only"""
        with open(filename, 'w') as f:
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    f.write("{:30} {}".format(a, getattr(self, a)))
                    f.write("\n")

    def to_dict(self):
        """To dictionary"""
        data = dict()
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                data[a] = getattr(self, a)
        return data

    def write_json(self, filename):
        """Write Configuration values. cannot be reloaded."""
        data = self.to_dict()
        data_series = pd.Series(data)
        data_series.to_json(filename)

    def to_md5(self):
        data_bytes = self.to_string().encode()
        checksum = hashlib.md5(data_bytes).hexdigest()
        return checksum
