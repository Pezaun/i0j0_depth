from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adadelta, SGD, Adam


def yolo_separable_net():
    inputs = Input(shape=(416, 416, 3))
    net = Conv2D(32, 3, padding="same")(inputs)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = SeparableConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = SeparableConv2D(128, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(64, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(128, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = SeparableConv2D(256, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(128, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(256, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = SeparableConv2D(512, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(256, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(512, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(256, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(512, 3, padding="same")(net)
    net = BatchNormalization()(net)
    c16 = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c16)

    net = SeparableConv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(512, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(512, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = SeparableConv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    c16 = Conv2D(64, 1, padding="same")(c16)
    c16 = BatchNormalization()(c16)
    c16 = LeakyReLU()(c16)

    c16 = Reshape((13,13,256))(c16)

    net = Concatenate()([c16, net])

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(45, 1, padding="same")(net)

    net = Flatten()(net)

    model = Model(inputs, net)

    return model

def yolo_convolutional_net():
    inputs = Input(shape=(416, 416, 3))
    net = Conv2D(32, 3, padding="same")(inputs)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = Conv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = Conv2D(128, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)


    net = Conv2D(64, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    
    net = Conv2D(128, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = Conv2D(256, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(128, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(256, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(net)

    net = Conv2D(512, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(256, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(512, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(256, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(512, 3, padding="same")(net)
    net = BatchNormalization()(net)
    c16 = LeakyReLU()(net)
    net = MaxPooling2D(pool_size=(2,2), strides=(2,2))(c16)

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(512, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(512, 1, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    c16 = Conv2D(64, 1, padding="same")(c16)
    c16 = BatchNormalization()(c16)
    c16 = LeakyReLU()(c16)

    c16 = Reshape((13,13,256))(c16)

    net = Concatenate()([c16, net])

    net = Conv2D(1024, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    net = Conv2D(45, 1, padding="same")(net)

    net = Flatten()(net)

    model = Model(inputs, net)
    
    return model
