#!/usr/bin/env python
import cv2
import pickle
import numpy as np
import random
from collections import defaultdict

class DFDLoader:
    def __init__(self, INDEX_PATH, FEATURES_PATH=None):
        print "Creating Loader...",
        with open(INDEX_PATH, "r") as f:
            self.images_paths = f.read().splitlines()
        if FEATURES_PATH != None:
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
            im_key = self.images_paths[instances_index].split("/")[-1][:-4]

            im_data = self.letter_image(im_data, net_input_dim).astype(np.float32)
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

    def test_data_generator(self, batch, net_input_dim=(224,224,3)):
        X = np.zeros((batch, net_input_dim[0], net_input_dim[1], net_input_dim[2])).astype(np.float32)

        batch_index = 0
        instances_index = 0
        files_names = []
        while True:
            im_data = cv2.imread(self.images_paths[instances_index])            
            files_names += [self.images_paths[instances_index].split("/")[-1][:-4]]

            im_key  = self.images_paths[instances_index].split("/")[-1][:-4]
            im_data = cv2.resize(im_data, (net_input_dim[0], net_input_dim[1])).astype(np.float32)
            im_data /= 255.0
            im_data = im_data[...,::-1]

            batch_index += 1
            batch_index = batch_index % batch
            instances_index += 1

            X[batch_index] = im_data

            if instances_index == len(self.images_paths):
                instances_index = 0                

            if batch_index == 0:
                yield X, files_names
                files_names = []

    def letter_image(self, im, net_input_dim=(224,224,3)):
        w = net_input_dim[1]
        h = net_input_dim[0]
        im_out = np.ones(net_input_dim) * 127

        new_w = im.shape[1];
        new_h = im.shape[0];
        if w/float(im.shape[1]) < h/float(im.shape[0]):
            new_w = w;
            new_h = (im.shape[0] * w)/im.shape[1];
        else:
            new_h = h;
            new_w = (im.shape[1] * h)/im.shape[0];

        im_out[(h - new_h) / 2:new_h + (h - new_h) / 2,(w - new_w) / 2:new_w + (w - new_w) / 2,:] = cv2.resize(im, (new_w, new_h))
        return im_out

if __name__ == "__main__":
    list_path     = "/home/gabriel/datasets/X_Dataset_segmentation_3K_VOC/VOC2007/train.txt"
    features_path = "/home/gabriel/python_code/yolo_depthwise/features/features.dat"
    dfd = DFDLoader(list_path, features_path)
    # dfd = DFDLoader(list_path)
    gen = dfd.data_generator(10, shuffle=True, net_input_dim=(416, 416, 3))
    # gen = dfd.test_data_generator(10)

    for i in range(1000):
        data = gen.next()
        print data[0].shape, data[1].shape, data[0].mean()
