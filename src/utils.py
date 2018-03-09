import os, sys, shutil
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from PIL import Image

from sklearn.model_selection import train_test_split

MASK_RNN_DIR = os.path.expanduser('~/Dropbox/lib/')
sys.path.append(MASK_RNN_DIR)
import Mask_RCNN.model as mrnn_modellib
import Mask_RCNN.utils as mrnn_utils
import Mask_RCNN.visualize as visualize

from .dataset import NucleiDataset

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

def collect_subclasses_from_folders(subclass_folder, original_image_folder):
    """convert manually subclassed images into list.
    input_folder: the folder containing subfolders named as, e.g. "102_bw_small_bright1"
        each folder contains images belong to the same subclass.
    return:
        subclass_dict: {'102_bw_small_bright1': {'images':[...], 'name': 'bw_small_bright1', 'num_id':102}}
        subclass_df: dataframe output
    """
    subclass_dirs = [x for x in os.listdir(subclass_folder) if not x.startswith(".")]
    subclass_dict = dict()
    for subclass in subclass_dirs:
        subclass_dict[subclass] = {
            "num_id": int(subclass.split("_")[0]),
            "name": "_".join(subclass.split("_")[1:]),
            "images": [x.rstrip(".png") for x in os.listdir(os.path.join(subclass_folder, subclass)) if not x.startswith(".")]
        }
    all_subclass_images = sum([x['images'] for x in subclass_dict.values()], [])
    all_images = [x for x in os.listdir(original_image_folder) if not x.startswith('.')]
    assert set(all_subclass_images) == set(all_images), \
        "Images in manually subclassed folder do not match the original images."

    subclass_df = pd.DataFrame(sum([[{"subclass": key, "num_id": val['num_id'],
                                        "name":val['name'], 'image':x}
                                        for x in val['images']] for key, val in subclass_dict.items()], []))
    return subclass_dict, subclass_df


def generate_stratified_sample_ids(subclass_df, image_ids=None, test_ratio=0.15, seed=None):
    """train val split stratified by sample subtypes
    subclass_df: dataframe of in the format of collect_subclasses_from_folders() output.
    image_ids: a subset of images in subclass_df. If None, use all images.
    test_ratio: proportion of test/val set
    seed: random_seed
    return:
        train: image file names of train set
        test: image file names of test/val set
    """
    if not image_ids is None:
        assert set(image_ids).issubset(set(subclass_df.image))
        data_use = subclass_df[subclass_df.image.isin(image_ids)]
    else:
        data_use = subclass_df

    train, test = train_test_split(data_use.image, test_size=0.15,
                                  stratify=data_use.num_id, random_state=seed)
    return list(train.values), list(test.values)


def move_only_images_to_folder(input_folder, output_folder):
    """Copy images from the orginal nested dataset folder to another folder.
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if len([x for x in os.listdir(output_folder) if not x.startswith(".")]) > 0:
        print("Target folder not empty. Abort.")
        return None
    subfolders = [x for x in os.listdir(input_folder) if not x.startswith(".")]
    for s in subfolders:
        image_folder = os.path.join(input_folder, s, 'images')
        print(s + ".png")
        shutil.copyfile(os.path.join(image_folder, s + ".png"), os.path.join(output_folder, s + ".png"))
    print(format("Copied %d images to folder: %s" % (len(subfolders), output_folder)))
    return

def move_subclass_to_folder(input_folder, output_folder, subclass_df,
                             image_ids=None, copy=True, postfix='.png'):
    """Move images to a folder in which the subfolders contain subclasses of images.
    input_folder: the folder (no subfolder) containing images to move
    output_folder: the target folder
    subclass_df: dataframe of in the format of collect_subclasses_from_folders() output.
    image_ids: a subset of images in subclass_df. If None, use all images.
    copy: whether to copy or move. (False is not yet implemented.)
    return:
        None
    """
    all_images = [x.rstrip(postfix) for x in os.listdir(input_folder) if x.endswith(postfix)]
    if not image_ids is None:
        assert set(image_ids).issubset(set(subclass_df.image))
        assert set(image_ids).issubset(set(all_images))
        data_use = subclass_df[subclass_df.image.isin(image_ids)]
    else:
        data_use = subclass_df[subclass_df.image.isin(all_images)]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if len([x for x in os.listdir(output_folder) if not x.startswith(".")]) > 0:
        print("Target folder not empty. Abort.")
        return None
    for i, row in data_use.iterrows():
        output_subfolder = os.path.join(output_folder, row['subclass'])
        if not os.path.exists(output_subfolder):
            os.mkdir(output_subfolder)
        shutil.copyfile(os.path.join(input_folder, row['image'] + postfix),
                        os.path.join(output_subfolder, row['image'] + postfix))
    print(format("%d files were copied from %s to %s" % (len(data_use), input_folder, output_folder)))
    return None

def combine_horizontal(images, scale = 1, same_size = True):
    # combine multiple PIL images
    if not same_size:
        min_height = min([x.size[1] for x in images])
        min_i = np.argmin([x.size[1] for x in images])
        scales = [min_height / x.size[1] for i, x in enumerate(images)]
        resized = images.copy()

        for i in range(len(resized)):
            if i != min_i:
                resized[i] = resized[i].resize([int(x * scales[i]) for x in resized[i].size], resample=Image.BICUBIC)
    else:
        resized = images

    width = sum([x.size[0] for x in resized])
    height = max([x.size[1] for x in resized])
    combined = Image.new('RGB', (width, height), (255,255,255))

    x_offset = 0
    for im in resized:
        if len(im.split()) > 3:
            combined.paste(im, (x_offset,0), mask=im.split()[3])
        else:
            combined.paste(im, (x_offset,0))
        x_offset += im.size[0]
    if scale != 1:
        combined = combined.resize([int(x * scale) for x in combined.size], resample=Image.BICUBIC)

    return combined

def generate_ground_truth(dataset_dir, dataset_name, output_folder,
                          image_ids=None, mask_alpha=0, resize=False):
    """produce ground truth mask images from labeled data.
    dataset_name: a folder inside dataset_dir
            each example (images + masks) are in a separate folder named after the image.
    output_folder:
    image_ids: subset of image id's to use
    mask_alpha: higher alpha gives stronger color to mask on top of raw image.
    """
    input_folder = os.path.join(dataset_dir, dataset_name)
    postfix = ".png"
    empty_classnames = ['','']
    all_images = [x for x in os.listdir(input_folder) if not x.startswith(".")]
    if not image_ids is None:
        assert set(image_ids).issubset(set(all_images))
        images_use = image_ids
    else:
        images_use = all_images

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if len([x for x in os.listdir(output_folder) if not x.startswith(".")]) > 0:
        print("Target folder not empty. Abort.")
        return None
    dataset = NucleiDataset()
    dataset.load_dataset(dataset_dir, dataset_name, image_ids = images_use)
    dataset.prepare()
    for i, image_id in enumerate(dataset.image_ids):
        if i % 50 == 0:
            print(i)
        original_image = dataset.load_image(image_id)
        gt_mask, gt_class_id = dataset.load_mask(image_id)
        gt_bbox = mrnn_utils.extract_bboxes(gt_mask)

        hw_ratio = original_image.shape[0] / original_image.shape[1]
        fig, axes = plt.subplots(1,2, figsize=(20,10 * hw_ratio))

        visualize.display_instances(original_image, np.array([]), np.array([]), np.array([]),
                                    empty_classnames, ax=axes[0], show=False, mask_alpha=0, verbose=False)
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    empty_classnames, ax=axes[1], show=False, mask_alpha=0, verbose=False)
        axes[1].set_title(format("ground truth: %d nuclei" % len(gt_class_id)), fontsize=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(output_folder, dataset.image_info[image_id]['id'] + postfix), dpi=100)
        plt.close(fig)
        gc.collect()
    print(format("%d ground truth images generated in folder %s" % (len(dataset.image_ids), output_folder)))
    return




def generate_detection_masks(dataset, results, output_folder, ground_truth=True, avg_precisions=None, image_ids=None, fill_holes=True):
    """produce detection masks from nuclei detection results. White image files to output_folder
    dataset:
    detection_result: the result coming from running detection on the input dataset.

    """
    image_ids_use = results.keys()
    assert set(image_ids_use).issubset(dataset.image_ids)
    if not avg_precisions is None:
        assert set(avg_precisions.keys()) == set(results.keys())
    postfix = ".png"
    empty_classnames = ['','']
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if len([x for x in os.listdir(output_folder) if not x.startswith(".")]) > 0:
        print("Target folder not empty. Abort.")
        return None
    for i, image_id in enumerate(image_ids_use):

        original_image = dataset.load_image(image_id)


        r = results[image_id]
        hw_ratio = original_image.shape[0] / original_image.shape[1]
        if ground_truth:
            gt_mask, gt_class_id = dataset.load_mask(image_id, fill_holes=True)
            gt_bbox = mrnn_utils.extract_bboxes(gt_mask)
            fig, axes = plt.subplots(1, 3, figsize=(30,10 * hw_ratio))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(20,10 * hw_ratio))
        visualize.display_instances(original_image, np.array([]), np.array([]), np.array([]),
                                    empty_classnames, ax=axes[0], show=False, mask_alpha=0, verbose=False)
        if ground_truth:
            visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                        empty_classnames, ax=axes[1], show=False, mask_alpha=0, verbose=False)
            axes[1].set_title(format("ground truth: %d nuclei" % len(gt_class_id)), fontsize=20)
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    empty_classnames, r['scores'], ax=axes[-1], show=False, mask_alpha=0, verbose=False)
        avg_prec_str = ""
        if not avg_precisions is None:
            avg_prec_str = format("AP: %s" % avg_precisions[image_id])
        axes[-1].set_title(format("detection: %d nuclei. %s" % (len(r['class_ids']), avg_prec_str)), fontsize=20)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, dataset.image_info[image_id]['id'] + postfix), dpi=100)
        plt.close(fig)
        gc.collect()
    print(format("%d detection images generated in folder %s" % (len(image_ids_use), output_folder)))
    return

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
        return 0.0, [0.0] * len(iou_thresholds), np.array([])
    if np.min(gt_masks.shape) == 0:
        print('ground truth mask is empty.')
        return 0.0, [0.0] * len(iou_thresholds), np.array([])

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
        return 0.0, [0.0] * len(iou_thresholds), np.array([])
    if np.min(gt_masks.shape) == 0:
        print('ground truth mask is empty.')
        return 0.0, [0.0] * len(iou_thresholds), np.array([])
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


def mAP_dsb2018(dataset, pred_results, image_ids, config=None, resize=False,
                iou_threshold=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                precision_func=nuclei_precision_simple, mask_overlaps_func=mask_overlaps_rle2):
    AP_all = dict()
    precisions_all = dict()
    overlaps_all = dict()
    for image_id in image_ids:
        # Load image and ground truth data
        if resize is True:
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                mrnn_modellib.load_image_gt(dataset, config,
                                       image_id, use_mini_mask=False)
        else:
            gt_mask, gt_class_id = dataset.load_mask(image_id, fill_holes=True)

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


def rle_encode_all(dataset, pred_results, image_ids, with_shape=True):
    rle_list = []
    for i, image_id in enumerate(image_ids):
        ori_id = dataset.image_info[image_id]['id']
        r = pred_results[image_id]
        if min(r['masks'].shape) == 0:
            continue
        for k in range(r['masks'].shape[2]):
            mask = r['masks'][:,:,k]
            rle = rle_encode(mask)
            if len(rle) == 0:
                continue
            rle_row = {'ImageId': ori_id, 'EncodedPixels': rle, 'Shape': format('%s %s' % mask.shape)}
            rle_list.append(rle_row)
    rle_df = pd.DataFrame(rle_list, columns=['ImageId', 'EncodedPixels', 'Shape'])
    if not with_shape:
        rle_df = rle_df[['ImageId', 'EncodedPixels']]
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
