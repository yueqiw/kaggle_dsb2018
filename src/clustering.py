import os, json
import sys
import random
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
import copy
import shutil


import os, sys
import numpy as np
import pandas as pd
from imagecluster import main
from imagecluster import imagecluster
from scipy.spatial import distance
from scipy.cluster import hierarchy


def cluster_auto(folder_path, cutoff=0.5):
    main.main(folder_path, sim=cutoff)

def get_embeddings(folder_path):
    with open(os.path.join(folder_path, "imagecluster/fingerprints.pk"), 'rb') as f:
        fp = pickle.load(f)
    return fp

def get_clusters_and_centers(fp, metric='cosine', cutoff=0.5):
    cluster_list = imagecluster.cluster(fp, metric="cosine", sim=cutoff)
    centers = [np.mean(np.array([fp[x] for x in c]), axis=0) for c in cluster_list]
    n_imgs = [len(x) for x in cluster_list]
    seq = np.argsort(n_imgs)[::-1]
    cluster_list = [cluster_list[i] for i in seq]
    centers = [centers[i] for i in seq]
    return cluster_list, centers

def distance_matrix(centers):
    dist = distance.pdist(np.array(centers), metric = "cosine")
    dist = distance.squareform(dist)
    return dist

def make_links_with_name(clusters, cluster_dr, id_start=100):
    # make the symlink names more intuitive.
    # group all clusters (cluster = list_of_files) into separate folders
    cdct = {}
    print("cluster dir: {}".format(cluster_dr))
    if os.path.exists(cluster_dr):
        shutil.rmtree(cluster_dr)
    for i, cls in enumerate(clusters):
        dr = os.path.join(cluster_dr, '{:0>3}_with_{}'.format(i + id_start, len(cls)))
        for fn in cls:
            link = os.path.join(dr, os.path.basename(fn))
            os.makedirs(os.path.dirname(link), exist_ok=True)
            os.symlink(os.path.abspath(fn), link)

def merge_clusters(fp, init_cluster_list, n_clusters_keep=11, max_dist=None, verbose=True):
    cluster_list = copy.deepcopy(init_cluster_list)
    cluster_list = sorted(cluster_list, key=len, reverse=True)
    for i in range(len(cluster_list) - n_clusters_keep):
        centers = [np.mean(np.array([fp[x] for x in c]), axis=0) for c in cluster_list]
        dist_mtx = distance_matrix(centers)
        dist_mtx[dist_mtx==0] = np.inf
        min_dist = dist_mtx.min(1)
        nbr_cluster = dist_mtx.argmin(1)
        nbr = nbr_cluster[-1]
        print('{}->{}'.format(len(cluster_list), nbr))
        cluster_list[nbr] += cluster_list[-1]
        cluster_list = cluster_list[:-1]
    return cluster_list

def print_cluster_nbr(fp, cluster_list):
    n_imgs = [len(x) for x in cluster_list]
    centers = [np.mean(np.array([fp[x] for x in c]), axis=0) for c in cluster_list]
    dist_mtx = distance_matrix(centers)
    dist_mtx[dist_mtx==0] = np.inf
    min_dist = dist_mtx.min(1)
    nbr_cluster = dist_mtx.argmin(1)
    print("\n".join(["{} (n={}): {}, dist={:.3f}".format(i, n_imgs[i], nbr_cluster[i], min_dist[i]) for i in range(len(min_dist))]))
