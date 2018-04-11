from __future__ import print_function
import os, sys
from itertools import product
import random
#import torch.multiprocessing as mp
#mp.set_start_method('forkserver')

FastPhotoStyleDir = os.path.expanduser('~/Dropbox/lib/FastPhotoStyle')
sys.path.append(FastPhotoStyleDir)

import argparse
import gc

import torch

import process_stylization
from photo_wct import PhotoWCT

MODEL_PATH = os.path.join(FastPhotoStyleDir, 'PhotoWCTModels/photo_wct.pth')

class PhotoStyle():
    def __init__(self, model_path=MODEL_PATH, use_cuda=True):
        self.p_wct = PhotoWCT()
        self.p_wct.load_state_dict(torch.load(model_path))
        self.use_cuda = use_cuda
        if use_cuda:
            self.p_wct.cuda(0)

    def stylize(self, content_image_path, style_image_path, output_image_path,
                content_seg_path=None, style_seg_path=None, smooth=True, verbose=False):
        process_stylization.stylization(
            p_wct=self.p_wct,
            content_image_path=content_image_path,
            style_image_path=style_image_path,
            content_seg_path=content_seg_path,
            style_seg_path=style_seg_path,
            output_image_path=output_image_path,
            cuda=self.use_cuda,
            smooth=smooth,
            verbose=verbose,
        )


def transfer(model, content_dir, style_dir, content, style, output_dir,
            with_masks=True, content_mask_dir=None, style_mask_dir=None,
            use_cuda=True, smooth=True, verbose=False):
    content_image = os.path.join(content_dir, content + '.png')
    style_image = os.path.join(style_dir, style + '.png')
    # note if a image is both content and style, the pair is skipped
    if with_masks:
        content_mask = os.path.join(content_mask_dir, content + '_mask.png')
        style_mask = os.path.join(style_mask_dir, style + '_mask.png')
    else:
        content_mask = style_mask = None
    seg_str = "withseg" if with_masks else "noseg"
    smooth_str = "sm" if smooth else "nosm"
    filename = format("stylize_%s_%s_%s_%s.png" % (seg_str, smooth_str, content, style))
    output_path = os.path.join(output_dir, filename)
    if verbose:
        print(filename, end='\n')
    try:
        model.stylize(content_image_path=content_image, style_image_path=style_image,
                output_image_path=output_path,
                content_seg_path=content_mask, style_seg_path=style_mask,
                smooth=smooth, verbose=verbose)
    except:
        print(format("Error occured for %s" % output_path))

def style_transfer(content_dir, style_dir, content_list, style_list,
                   output_dir, with_masks=True, content_mask_dir=None, style_mask_dir=None,
                   max_output=None, use_cuda=True, smooth=True, num_processes=1,
                   verbose=False):
    """Batch style transfer.
    content_dir:
    style_dir:
    content_list: list of file names without postfix (.png)
    style_list:
    output_dir:
    with_masks:
    subsample: int
    """

    content_style_pairs = list(product(content_list, style_list))

    # subsample in case there are too many images
    n_iters = int(len(content_style_pairs)/num_processes) + 1
    if max_output and max_output < len(content_style_pairs):
        content_style_pairs = random.sample(content_style_pairs, max_output)  # list
    n_pairs = len(content_style_pairs)

    # convert to generator
    content_style_pairs = enumerate(content_style_pairs)

    photo_style = PhotoStyle(model_path=MODEL_PATH, use_cuda=use_cuda)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if num_processes == 1:
        for i, (content, style) in content_style_pairs:
            print(format("Saving %d/%d outputs to %s" % (i, n_pairs, output_dir)),
                    end="\r")
            #print("%d - " % i, end="")
            kwargs = {'with_masks': with_masks, \
                'content_mask_dir': content_mask_dir, 'style_mask_dir': style_mask_dir, \
                'use_cuda': use_cuda, 'smooth': smooth, 'verbose': verbose}
            transfer(photo_style, content_dir, style_dir, content, style, \
                output_dir, **kwargs)

    # Hell using multi-process is actually slower than single process
    # Because when using single process, all CPU cores are at high usage
    # i guess the bottleneck is on CPU side rather than GPU.
    # don't use it
    elif num_processes > 1:
        photo_style.p_wct.share_memory()
        for _ in range(n_iters):
            processes = []
            for rank in range(num_processes):
                i, (content, style) = next(content_style_pairs, (None, (None, None)))
                if i is None:
                    break
                print(format("Saving %d/%d outputs to %s" % (i + 1, n_pairs, output_dir)),
                        end="\r")
                p = mp.Process(target=transfer,
                    args=(photo_style, content_dir, style_dir, content, style, \
                        output_dir),
                    kwargs = {'with_masks': with_masks, \
                        'content_mask_dir': content_mask_dir, 'style_mask_dir': style_mask_dir, \
                        'use_cuda': use_cuda, 'smooth': smooth, 'verbose': verbose})
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
    print("\n")
    #del photo_style
    #gc.collect()
    #torch.cuda.empty_cache()
    return
