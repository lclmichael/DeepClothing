# encoding=utf8
# Author=LclMichael

import os

import tensorflow as tf

from deepclothing.util import json_utils
from deepclothing.util import image_utils
from deepclothing.util import config_utils

def decode_original_image(image, label):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [224, 224])
    label = tf.one_hot(label, 50)
    label = tf.cast(label, dtype=tf.float32)
    return image, label

def get_iterator(tensors, batch_size=32, threads=8, num_epochs=-1, is_shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.map(decode_original_image, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    if is_shuffle:
        dataset = dataset.shuffle(batch_size * threads)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

class PredictionReader(object):
    # deepfashion root dir
    _base_dir = config_utils.get_global("deepfashion_root_dir")

    _image_dir = "Category and Attribute Prediction Benchmark/Img/"

    _json_dir = "./json/"

    def set_dir(self, base_dir=None, json_dir=None):
        if base_dir is not None:
            self._base_dir = base_dir
        if json_dir is not None:
            self._json_dir = json_dir

    # get batch from json, return a tenor list [img_batch, label_batch]
    def get_batch_from_json(self, json_name, batch_size=32, is_shuffle=True):
        data = json_utils.read_json_file(os.path.join(self._json_dir, json_name))
        img_list = []
        label_list = []
        for item in data:
            img_list.append(os.path.join(self._base_dir, self._image_dir, item["path"]))
            label_list.append(item["categoryNum"])
        tensors = (img_list, label_list)
        batch = get_iterator(tensors, batch_size=batch_size, is_shuffle=is_shuffle)
        return batch

    def test_batch(self):
        train_json_file = "prediction_train.json"
        batch_size = 32
        batch = self.get_batch_from_json(train_json_file, batch_size)
        with tf.Session() as sess:
            img_batch, label_batch = sess.run(batch)
            for i in range(batch_size):
                print(label_batch[i])
                image_utils.show_image(img_batch[i])

def main():
    pr = PredictionReader()
    pr.test_batch()

if __name__ == '__main__':
    main()


