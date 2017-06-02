#!/usr/bin/env python
import numpy as np
import pickle

if __name__ == "__main__":
    YOLO_FEATURES_PATH = "/home/gabriel/python_code/yolo_depthwise/features/features.dat"
    DW_FEATURES_PATH   = "/home/gabriel/python_code/yolo_depthwise/preds/preds_conv_35dec.dat"

    with open(YOLO_FEATURES_PATH, "r") as f:
        yolo_features_dic = pickle.load(f)

    with open(DW_FEATURES_PATH, "r") as f:
        dw_features_tup   = pickle.load(f)

    print type(dw_features_tup)