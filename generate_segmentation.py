import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import cv2
from multiprocessing import Pool,Value

sample_name = "sample1_frames"
save_fol = "./results/segmentation/"
path_dir = '/home/pranjali/Documents/AnomalyDetection/sample_seq/' + sample_name + '/'

path_dir = '/media/god-particle/DA48D0B148D08D9F/shaantanu_honours/data/color/'
# path_dir = './'


def generate_segmentation(file):
    global curr_counter
    with curr_counter.get_lock():
        print(f"Starting segmentation for {file} , {curr_counter.value} \n")
        curr_counter.value += 1
    img = Image.open(path_dir + file)
    img = np.array(img)
    shape = img.shape
    reshaped_img = np.reshape(img, [-1,3])
    bandwidth = estimate_bandwidth(reshaped_img, quantile=0.1, n_samples=100)
    msc = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    nsamples, nx, ny = img.shape
    msc.fit(reshaped_img)
    labels = msc.labels_
    seg_image = np.reshape(labels, shape[:2])

    cv2.imwrite(save_fol+file , seg_image)

    print(f"save done for {file} \n")

if __name__ == "__main__":
    filelist= [file for file in os.listdir(path_dir) if file.endswith('.png')]
    filelist.sort()

    curr_counter = Value('i',0)


    with Pool() as pool:
        pool.map(generate_segmentation,filelist)



