import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

def load_images(im_dir, size=(256, 512)):
    src_list, tar_list = list(), list()
    for file_name in os.listdir(im_dir):
        pixels = load_img(im_dir + file_name, target_size=size)
        pixels = img_to_array(pixels)
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [np.asarray(src_list), np.asarray(tar_list)]

def compress_imdir_npz(im_dir, npz_name):
    [src_ims, tar_ims] = load_images(im_dir)
    print("Dataset Loaded: ", src_ims.shape, tar_ims.shape)
    np.savez_compressed(npz_name, src_ims, tar_ims)
    print("Dataset Saved: ", npz_name)
