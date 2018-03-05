import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils

MASK_RNN_DIR = os.path.expanduser('~/Dropbox/lib/')
sys.path.append(MASK_RNN_DIR)
import Mask_RCNN.model as mrnn_modellib
import Mask_RCNN.utils as mrnn_utils

def generate_sample_ids(dataset_dir, split_type='random', val_ratio=0.15, seed=0):
    train_all_ids = [x for x in os.listdir(dataset_dir) if not x.startswith('.')]
    n_samples = len(train_all_ids)
    if split_type == 'random':
        np.random.seed(seed)
        shuffled_ids = [train_all_ids[x] for x in np.random.permutation(len(train_all_ids))]
        n_val = int(n_samples * val_ratio)
        n_train = n_samples - n_val
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:]
        print(format("Total %d samples. Train: %d, Val: %d" % (n_samples, n_train, n_val)))
        return train_ids, val_ids
    else:
        print(format("spilt_type %s not implemented." % split_type))



def mask_overlaps(pred_masks, gt_masks):
    pred_masks_flat = pred_masks.reshape((-1, pred_masks.shape[2]))  # m2 x p
    gt_masks_flat = gt_masks.reshape((-1, gt_masks.shape[2]))  # m2 x g
    intersection = np.dot(pred_masks_flat.T, gt_masks_flat)  # p * g

    pred_masks_sum = pred_masks.sum(axis=(0,1))
    gt_masks_sum = gt_masks.sum(axis=(0,1))
    mask_sum_grid = np.meshgrid(gt_masks_sum, pred_masks_sum)  # p * g
    union = mask_sum_grid[0] + mask_sum_grid[1] - intersection
    iou = intersection / union.astype('float')
    return iou

def mask_overlaps_rle(pred_masks, gt_masks):
    overlaps = np.zeros((pred_masks.shape[2], gt_masks.shape[2]))
    for i in range(overlaps.shape[0]):
        for j in range(overlaps.shape[1]):
            pred_rle = maskUtils.encode(np.asfortranarray(pred_masks[:,:,i]))
            #print(pred_rle)
            gt_rle = maskUtils.encode(np.asfortranarray(gt_masks[:,:,j]))
            #print(gt_rle)
            #print(maskUtils.area(gt_rle))
            #print([int(o['iscrowd']) for o in gt_rle])
            overlaps[i, j] = maskUtils.iou([pred_rle], [gt_rle], [0])
    return overlaps

def mask_overlaps_rle2(pred_masks, gt_masks):
    pred_rle = [maskUtils.encode(np.asfortranarray(pred_masks[:,:,i])) for i in range(pred_masks.shape[2])]
    gt_rle = [maskUtils.encode(np.asfortranarray(gt_masks[:,:,i])) for i in range(gt_masks.shape[2])]
    iscrowd = [0 for i in gt_rle]
    ious = maskUtils.iou(pred_rle,gt_rle,iscrowd)
    return ious

def mask_iou_slow(pred_mask, gt_mask):
    """
    pred_mask: 2D array
    gt_mask: 2D array of same size
    """
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
    iou = intersection / union.astype('float')
    return iou

def mask_overlaps_slow(pred_masks, gt_masks):
    overlaps = np.zeros((pred_masks.shape[2], gt_masks.shape[2]))
    for i in range(overlaps.shape[0]):
        for j in range(overlaps.shape[1]):
            overlaps[i, j] = mask_iou_slow(pred_masks[:,:,i], gt_masks[:,:,j])
    return overlaps



def nuclei_precision_simple(gt_masks, gt_class_ids,
                     pred_masks, pred_class_ids, pred_scores,
                     iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                     mask_overlaps_func=mask_overlaps_rle2):
    # this only works when min(iou_thresholds) >= 0.5, otherwise use nuclei_precision_complex()
    # note some variables are never used.
    if np.min(pred_masks.shape) == 0:
        print('prediction mask is empty.')
        return 0.0, None, None
    if np.min(gt_masks.shape) == 0:
        print('ground truth mask is empty.')
        return 0.0, None, None

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = mask_overlaps_func(pred_masks, gt_masks)

    max_pred_iou = np.max(overlaps, axis=1)

    prec_list = []
    for iou_threshold in iou_thresholds:
        match_count = np.count_nonzero(max_pred_iou > iou_threshold)
        # precision = TP/(TP + FP + FN)
        precision = match_count / float(overlaps.shape[0] + overlaps.shape[1] - match_count)
        prec_list.append(precision)

    mAP = np.mean(prec_list)
    return mAP, prec_list, overlaps


def nuclei_precision_complex(gt_masks, gt_class_ids,
                     pred_masks, pred_class_ids, pred_scores,
                     iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                     mask_overlaps_func=mask_overlaps_rle2):
    if np.min(pred_masks.shape) == 0:
        print('prediction mask is empty.')
        return 0.0, None, None
    if np.min(gt_masks.shape) == 0:
        print('ground truth mask is empty.')
        return 0.0, None, None
    pred_scores = pred_scores[:pred_masks.shape[2]]
    indices = np.argsort(pred_scores)[::-1]
    pred_masks = pred_masks[:,:,indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = mask_overlaps_func(pred_masks, gt_masks)

    prec_list = []
    for iou_threshold in iou_thresholds:
        # Loop through ground truth boxes and find matching predictions
        match_count = 0
        pred_match = np.zeros(pred_masks.shape[2])
        gt_match = np.zeros(gt_masks.shape[2])
        for i in range(pred_masks.shape[2]):
            # Find best matching ground truth mask
            # since the lowest iou threshold is 0.5, and masks are non-overlapping
            # matching between predictions and ground truth is unique
            # either the one with highest iou either match or none of them match
            # the following code might be useful for iou threshold < 0.5 and/ multiclass

            sorted_ixs = np.argsort(overlaps[i])[::-1]
            for j in sorted_ixs:
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_threshold:
                    # print('match not found.')
                    break
                # If ground truth box is already matched, go to next one
                if gt_match[j] == 1:
                    print('already matched.')
                    continue
                # Do we have a match?
                if pred_class_ids[i] == gt_class_ids[j]:
                    # print('matched.')
                    match_count += 1
                    gt_match[j] = 1
                    pred_match[i] = 1
                    break
        # precision = TP/(TP + FP + FN)
        precision = match_count / float(len(pred_match) + sum(gt_match == 0))
        # same result using the formula below
        # precision_2 = match_count / float(pred_masks.shape[2] + gt_masks.shape[2] - match_count)
        prec_list.append(precision)

    mAP = np.mean(prec_list)

    return mAP, prec_list, overlaps

def mAP_VOC_bbox(dataset, pred_results, image_ids, inference_config, iou_threshold=0.5):
    # Compute VOC-Style mAP
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            mrnn_modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)

        r = pred_results[image_id]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            mrnn_utils.compute_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"],
                             iou_threshold=iou_threshold)
        APs.append(AP)
    return(np.mean(APs), APs, precisions, overlaps)


def mAP_dsb2018(dataset, pred_results, image_ids, inference_config, resize=True,
                iou_threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                precision_func=nuclei_precision_simple, mask_overlaps_func=mask_overlaps_rle2):
    AP_all = dict()
    precisions_all = dict()
    overlaps_all = dict()
    for image_id in image_ids:
        # Load image and ground truth data
        if resize is True:
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                mrnn_modellib.load_image_gt(dataset, inference_config,
                                       image_id, use_mini_mask=False)
        else:
            gt_mask, gt_class_id = dataset.load_mask(image_id)

        r = pred_results[image_id]
        # Compute AP
        AP, precisions, overlaps =\
        precision_func(gt_mask, gt_class_id,
                         r["masks"], r["class_ids"], r["scores"],
                         mask_overlaps_func=mask_overlaps_func)
        AP_all[image_id] = AP
        precisions_all[image_id] = precisions
        overlaps_all[image_id] = overlaps
    return(np.mean(list(AP_all.values())), AP_all, precisions_all, overlaps_all)


def remove_overlap(pred_masks, pred_scores):
    overlaps = mask_overlaps_rle2(pred_masks, pred_masks)

    for i in range(overlaps.shape[0]):
        for j in range(overlaps.shape[0]):
            if i > j and overlaps[i, j] > 0:
                intersection = pred_masks[:,:,i] * pred_masks[:,:,j]
                if pred_scores[i] > pred_scores[j]:
                    pred_masks[:,:,j] -= intersection
                else:
                    pred_masks[:,:,i] -= intersection

    return pred_masks, overlaps




def nb_get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def rle_encode_all(dataset, pred_results, image_ids):
    rle_list = []
    for i, image_id in enumerate(image_ids):
        ori_id = dataset.image_info[image_id]['id']
        r = pred_results[image_id]
        for k in range(r['masks'].shape[2]):
            mask = r['masks'][:,:,k]
            rle = rle_encode(mask)
            rle_row = {'ImageId': ori_id, 'EncodedPixels': rle, 'Shape': format('%s %s' % mask.shape)}
            rle_list.append(rle_row)
    rle_df = pd.DataFrame(rle_list, columns=['ImageId', 'EncodedPixels', 'Shape'])
    return rle_df

def rle_decode_all(dataset, rle_df):
    mask_results = dict()
    for image_id_str in rle_df['ImageId'].unique():
        rle_subset = rle_df[rle_df['ImageId']==image_id_str]
        mask_list = []
        for i, row in rle_subset.iterrows():
            shape = tuple(map(int, row['Shape'].split()))
            mask_rle = row['EncodedPixels']
            mask = rle_decode(mask_rle, shape)
            mask_list.append(mask)
        mask_all = np.stack(mask_list, axis=2)
        class_ids = np.ones(len(mask_list))
        scores = np.ones(len(mask_list), dtype='float')
        int_id = dataset.get_int_id(image_id_str)
        mask_results[int_id] = {
            'class_ids': class_ids,
            'masks': mask_all,
            'scores': scores
        }
    return mask_results



def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten(order='F')

    # add this to fix edge conditions
    px_first = pixels[0]
    px_last = pixels[-1]

    # the following two lines causes differences with dsb2018 example file. Thus add above lines.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    if px_first > 0:
        runs[0] -= 1
        runs[1] += 1
    if px_last > 0:
        runs[-1] += 1
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
