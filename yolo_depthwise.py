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
    model = nets.yolo_convolutional_net()
    opt = Adam(lr=1e-3)
    # opt = SGD(lr=1e-3, momentum=0.9, decay=5e-4, nesterov=False)
    model.compile(optimizer=opt, loss="mse")
    model.summary()

    train_list_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K_VOC/VOC2007/train.txt"
    valid_list_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K_VOC/VOC2007/valid.txt"
    features_path   = "/home/gabriel/python_code/yolo_depthwise/features/features.dat"

    MODEL_PATH     = "/home/gabriel/python_code/yolo_depthwise/models_6dec/weights.{epoch:02d}-{val_loss:.8f}.hdf5"
    LOG_PATH       = "/home/gabriel/python_code/yolo_depthwise/logs_6dec/training.log"

    dfd_train = DFDLoader(train_list_path, features_path)
    dfd_valid = DFDLoader(valid_list_path, features_path)

    data_train = dfd_train.data_generator(8, shuffle=True,  net_input_dim=(416, 416, 3))
    data_valid = dfd_valid.data_generator(10, shuffle=False, net_input_dim=(416, 416, 3))

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    logger     = CSVLogger(LOG_PATH)
    lr_schedul = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

    model.fit_generator(data_train, epochs=500, steps_per_epoch=264, validation_data=data_valid, validation_steps=30, callbacks=[checkpoint, logger, lr_schedul])
