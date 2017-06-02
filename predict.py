#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adadelta, SGD, Adam
from loader import DFDLoader
import nets
import pickle

# Workaround for not using all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

if __name__ == "__main__":
    model = nets.yolo_separable_net()
    model.summary()
    print "Loading..."
    model.load_weights("/home/gabriel/python_code/yolo_depthwise/models/weights.15-2.15037974.hdf5")
    # print "Compiling..."
    # model.compile()
    print "Loading Data..."
    test_list_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K_VOC/VOC2007/test.txt"
    MODEL_PATH     = "/home/gabriel/python_code/yolo_depthwise/models/weights.15-2.15037974.hdf5"


    dfd_test = DFDLoader(test_list_path)
    
    data_test = dfd_test.test_data_generator(2, net_input_dim=(416, 416, 3))
    results = []
    print "Predicting..."
    for i in range(300):
        print i,
        result = model.predict_on_batch(data_test.next())
        results += [result[0]]
        results += [result[1]]
    with open("/home/gabriel/python_code/yolo_depthwise/preds.dat", "w") as f:
        pickle.dump(results, f)

