import os, sys
import skimage
import numpy as np
from pycocotools import mask as maskUtils
from skimage import morphology
from skimage.util import invert

MASK_RNN_DIR = os.path.expanduser('~/Dropbox/lib/')
sys.path.append(MASK_RNN_DIR)
import Mask_RCNN.utils as mrnn_utils

class NucleiDataset(mrnn_utils.Dataset):
    def __init__(self, class_map=None):
        super().__init__(class_map)
        self.id_mapping_str2int = dict()

    def load_dataset(self, dataset_dir, subset, image_ids=None, return_data=False):
        """Load a subset of the nuclei dataset_dir
        dataset_dir:
        subset: stage1_train, stage1_test
        """
        self.add_class(source = "dsb", class_id = 1, class_name = "nuclei")

        image_dir = os.path.join(dataset_dir, subset)
        image_all = [x for x in os.listdir(image_dir) if not x.startswith('.')]
        if not image_ids is None:
            assert set(image_ids).issubset(image_all)
        else:
            image_ids = image_all

        for i in image_ids:
            new_id = self.add_image(
                        source = "dsb",
                        image_id = i,
                        path = os.path.join(image_dir, i, 'images', i + '.png'),
                        width = None,
                        height = None,
                        annotations = None,
                        mask_dir = os.path.join(image_dir, i, 'masks')
                    )
            self.id_mapping_str2int[i] = new_id


    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        return(len(self.image_info) - 1)

    def load_mask(self, image_id, fill_holes=True):
        """
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "dsb":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        mask_dir = image_info['mask_dir']
        mask_path_list = [x for x in os.listdir(mask_dir) if not x.startswith('.')]
        for mask_path in mask_path_list:
            m = skimage.io.imread(os.path.join(mask_dir, mask_path))
            m[m.nonzero()] = 1
            if fill_holes:
                m = m.astype(bool)
                m = morphology.remove_small_holes(m, 64, connectivity=1, in_place=True)
                m = m.astype(np.uint8)
            instance_masks.append(m)
        class_ids = [1] * len(instance_masks)

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def load_image(self, image_id, invert_dark=True, dark_threshold=100):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim < 3:
            image = skimage.color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = image[:,:,:3]
        if invert_dark:
            avg_inten = np.sum(image) / np.product(image.shape)
            if avg_inten < dark_threshold:
                image = invert(image)

        return image

    def get_int_id(self, str_id):
        return self.id_mapping_str2int[str_id]

    def image_reference(self):
        pass
