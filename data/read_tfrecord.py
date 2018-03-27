# encoding=utf8
# Author="LclMichael"

import os

import tensorflow as tf
import matplotlib.pyplot  as plt

import deepclothing.util.image_utils as iu

RECORD_PATH = "./tfrecord/test.tfrecords"

def decode(serialize_example):
    features = tf.parse_single_example(
        serialize_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        }
    )

    image = tf.decode_raw(features["image_raw"], tf.float32)
    image.set_shape((300, 300, 3))
    label = tf.cast(features["label"], tf.int64)
    return image, label

def inputs(train, tfrecord_path, batch_size=5, num_epochs=1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)

def main():
    pass

if __name__ == '__main__':
    main()
