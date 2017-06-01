#!/usr/bin/env python
import cv2
import pickle
import numpy as np
import random
from collections import defaultdict

class DFDLoader:
    def __init__(self, INDEX_PATH, FEATURES_PATH):
        print "Creating Loader...",
        with open(INDEX_PATH, "r") as f:
            self.images_paths = f.read().splitlines()
        with open(FEATURES_PATH, "r") as f:
            self.features_dict = pickle.load(f)
        self.instance_count = len(self.images_paths)
        print "Done!"

    def data_generator(self, batch, shuffle=False, net_input_dim=(224,224,3)):
        X = np.zeros((batch, net_input_dim[0], net_input_dim[1], net_input_dim[2])).astype(np.float32)
        Y = np.zeros((batch, 7605)).astype(np.float32)

        batch_index = 0
        instances_index = 0
        while True:
            im_data = cv2.imread(self.images_paths[instances_index])            

            im_key  = self.images_paths[instances_index].split("/")[-1][:-4]
            im_data = cv2.resize(im_data, (net_input_dim[0], net_input_dim[1])).astype(np.float32)
            im_data /= 255.0
            im_data = im_data[...,::-1]

            batch_index += 1
            batch_index = batch_index % batch
            instances_index += 1

            X[batch_index] = im_data
            Y[batch_index] = self.features_dict[im_key]

            if instances_index == len(self.images_paths):
                instances_index = 0
                if shuffle:
                    random.shuffle(self.images_paths)

            if batch_index == 0:
                yield X, Y

if __name__ == "__main__":
    list_path     = "/home/gabriel/datasets/X_Dataset_segmentation_3K_VOC/VOC2007/train.txt"
    features_path = "/home/gabriel/python_code/yolo_depthwise/features.dat"
    dfd = DFDLoader(list_path, features_path)
    gen = dfd.data_generator(10, shuffle=True)

    for i in range(1000):
        data = gen.next()
        print "M:", data[0].mean(), "S:", data[0].std(), data[1].min(), data[1].max(), data[0].shape, data[1].shape
