# encoding=utf8
# Author="LclMichael"

import os

import tensorflow as tf

import deepclothing.util.image_utils as iu

def decode(serialize_example):
    features = tf.parse_single_example(
        serialize_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features["image_raw"], tf.float32)
    image = tf.reshape(image, [300, 300, 3])
    label = tf.cast(features["label"], tf.int64)
    return image, label

def augment(image, label):
    return image, label

def normalize(image, label):
    return image, label

def input_data(tfrecord_path, batch_size=5, num_epochs=1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(decode)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

class TFRecordReader(object):

    _tfrecord_dir = "./tfrecord/"

    def set_tfrecord_dir(self, dir_name):
        self._tfrecord_dir = dir_name

    def get_batch(self, batch_size=5):
        with tf.Session() as sess:
            data_batch = input_data(os.path.join(self._tfrecord_dir, "train.tfrecords"))
            count = 0
            while 1:
                image_batch_one, label_batch_one = sess.run(data_batch)
                for i in image_batch_one:
                    count += 1
                    print(count)


def main():
    tr = TFRecordReader()
    tr.get_batch()
    pass

if __name__ == '__main__':
    main()
