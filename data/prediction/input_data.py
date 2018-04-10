# encoding=utf8
# Author=LclMichael

import os

import numpy as np
import tensorflow as tf

from deepclothing.util import json_utils
from deepclothing.util import image_utils
from deepclothing.util import config_utils

def decode_original_image(image, label, bbox):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = tf.image.crop_to_bounding_box(image, y, x, height, width)
    image = tf.image.resize_images(image, [224, 224])
    label = tf.one_hot(label, 50)
    label = tf.cast(label, dtype=tf.float32)
    return image, label

def get_iterator(tensors, batch_size=32, threads=4, num_epochs=-1, is_shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.map(decode_original_image, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    if is_shuffle and batch_size >= 1:
        dataset = dataset.shuffle(batch_size * 2)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def get_data(name="train", batch_size=16, is_shuffle=True, data_root_dir=None, json_path=None):
    pr = PredictionReader()
    pr.set_dir(data_root_dir, json_path)
    train_batch = pr.get_batch_from_json("prediction_" + name + ".json", batch_size, is_shuffle=is_shuffle)
    return train_batch

class PredictionReader(object):
    # deepfashion root dir
    _data_root_dir = config_utils.get_global("deepfashion_root_dir")

    _image_dir = "Category and Attribute Prediction Benchmark/Img/"

    _json_dir = "./json/"

    _category_chs = "prediction_category_chs.json"

    _category_list = "prediction_category.json"

    def set_dir(self, data_root_dir=None, json_dir=None):
        if data_root_dir is not None:
            self._data_root_dir = data_root_dir
        if json_dir is not None:
            self._json_dir = json_dir

    def get_category_chs_list(self):
        file_path = os.path.join(self._json_dir, self._category_chs)
        data = json_utils.read_json_file(file_path)
        return data

    def get_category_list(self):
        file_path = os.path.join(self._json_dir, self._category_list)
        data = json_utils.read_json_file(file_path)
        return data

    # get batch from json, return a tenor list [img_batch, label_batch]
    def get_batch_from_json(self, json_name, batch_size=32, is_shuffle=True):
        data = json_utils.read_json_file(os.path.join(self._json_dir, json_name))
        if is_shuffle:
            np.random.shuffle(data)
        img_list = []
        label_list = []
        bbox_list = []
        for item in data:
            img_list.append(os.path.join(self._data_root_dir, self._image_dir, item["path"]))
            label_list.append(item["categoryNum"])
            bbox_list.append(item["bbox"])
        tensors = (img_list, label_list, bbox_list)
        if batch_size == -1:
            batch_size = len(img_list)
        batch = get_iterator(tensors, batch_size=batch_size, is_shuffle=is_shuffle)
        return batch

    def test_batch(self):
        batch_size = 16
        #中文名类别
        chs_list = self.get_category_chs_list()
        category_list = self.get_category_list()
        batch_tensor = get_data("test", batch_size=batch_size, is_shuffle=False)
        with tf.Session() as sess:
            for i in range(2500):
                img_batch, label_batch = sess.run(batch_tensor)
                print("image batch size: {}, label batch size: {}".format(len(img_batch), len(label_batch)))
                for j in range(batch_size):
                    index = np.argmax(label_batch[j])
                    print(index, chs_list[index], category_list[index])
                    image_utils.show_image(img_batch[j])
def main():
    pr = PredictionReader()
    # print(pr.get_category_chs_list())
    pr.test_batch()

if __name__ == '__main__':
    main()


