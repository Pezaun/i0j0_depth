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

# Workaround for not using all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

if __name__ == "__main__":
    model = nets.yolo_separable_net()
    model.summary()

    test_list_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K_VOC/VOC2007/test.txt"
    MODEL_PATH     = "/home/gabriel/python_code/yolo_depthwise/models/weights.15-2.15037974.hdf5"


    dfd_test = DFDLoader(test_list_path)
    
    data_test = dfd_test.data_generator(10, shuffle=False, net_input_dim=(416, 416, 3))
    
    model.fit_generator(data_train, epochs=500, steps_per_epoch=264, validation_data=data_valid, validation_steps=30, callbacks=[checkpoint, logger, lr_schedul])
