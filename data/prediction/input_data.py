# encoding=utf8
# Author=LclMichael

import os
import time

import numpy as np
import tensorflow as tf

import matplotlib.image as mpimage

from deepclothing.util import json_utils
from deepclothing.util import image_utils
from deepclothing.util import config_utils

global_mean = 100

global_variance = 100

#预处理
def image_preprocess(image, label, bbox):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = tf.image.crop_to_bounding_box(image, y, x, height, width)

    image = tf.image.resize_images(image, [224, 224])
    image = tf.subtract(image, global_mean)
    image = tf.div(tf.subtract(image, global_mean) ,global_variance)
    label = tf.one_hot(label, 50)
    return image, label

#求取均值的预处理
def mean_preprocess(image, bbox):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = tf.image.crop_to_bounding_box(image, y, x, height, width)
    image = tf.image.resize_images(image, [224, 224])
    #单张图的均值和方差
    moment = tf.nn.moments(image, [0,1,2,3])
    return moment

def get_iterator(tensors, batch_size=32, threads=4, num_epochs=-1, is_shuffle=False, preprocess=image_preprocess):
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.map(preprocess, num_parallel_calls=threads)
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

    def get_mean_without_tf(self, json_name):
        data = json_utils.read_json_file(os.path.join(self._json_dir, json_name))
        mean_list = []
        i = 0
        for item in data:
            i += 1
            name = os.path.join(self._data_root_dir, self._image_dir, item["path"])
            bbox = item["bbox"]
            x = bbox[0]
            y = bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            img = mpimage.imread(name)
            img = image_utils.crop_and_resize_image(img, (224, 224), y, x, height, width)
            mean = image_utils.get_image_mean(img)
            mean_list.append(mean)
            print("done for %d" % i)
        print(np.mean(np.array(mean_list).flatten(), 0))

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
                    print(img_batch[j])

                    image_utils.show_image(img_batch[j])


def main():
    pr = PredictionReader()
    # print(pr.get_category_chs_list())
    start = time.time()
    pr.get_mean_without_tf(json_name="prediction_train.json")
    print("cost time %d" % time.time() - start)

if __name__ == '__main__':
    main()


