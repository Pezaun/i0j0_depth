#!/usr/bin/env python
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.pooling import MaxPooling2D

if __name__ == "__main__":
    BATCH = 32
    inputs = Input(shape=(416, 416, 3))
    net = Conv2D(32, 3, padding="same", activation="relu")(inputs)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)
    net = SeparableConv2D(64, 3, padding="same", activation="relu")(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)
    net = SeparableConv2D(128, 3, padding="same", activation="relu")(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)
    net = SeparableConv2D(256, 3, padding="same", activation="relu")(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)
    net = SeparableConv2D(512, 3, padding="same", activation="relu")(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)
    net = SeparableConv2D(1024, 3, padding="same", activation="relu")(net)
    net = Conv2D(45, 3, padding="same", activation="relu")(net)

    model = Model(inputs, net)
    model.compile(optimizer="sgd", loss="mse")
    model.summary()
    print "ok!"
