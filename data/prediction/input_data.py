# encoding=utf8
# Author=LclMichael

import os
import time

import numpy as np
import tensorflow as tf

import deepclothing.data.prediction.json_data as json_data
from deepclothing.util import image_utils
from deepclothing.util import config_utils

train_bbox_mean = 157

train_mean = 185

#预处理
def preprocess_with_bbox(path, label, bbox):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = tf.image.crop_to_bounding_box(image, y, x, height, width)
    image = tf.image.resize_images(image, [224, 224])
    image = tf.subtract(image, train_bbox_mean)
    label = tf.one_hot(label, 50)
    return image, label

#preprocess image
def mean_preprocess_with_bbox(path, bbox):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = tf.image.crop_to_bounding_box(image, y, x, height, width)
    mean, variance = tf.nn.moments(image, [0, 1, 2])
    return mean

#预处理
def default_preprocess(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.subtract(image, train_bbox_mean)
    image = tf.image.resize_images(image, [224, 224])
    label = tf.one_hot(label, 50)
    return image, label

#求取均值的预处理
def mean_preprocess(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    # image = tf.image.resize_images(image, [224, 224])
    mean, variance = tf.nn.moments(image, [0, 1, 2])
    return mean

def get_iterator(datas, batch_size=32, threads=4, num_epochs=-1, is_shuffle=False, preprocess=default_preprocess):
    dataset = tf.data.Dataset.from_tensor_slices(datas)
    dataset = dataset.map(preprocess, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    if is_shuffle and batch_size >= 1:
        dataset = dataset.shuffle(batch_size * 2)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def get_data(name="train", batch_size=16, is_shuffle=True):
    pr = InputData()
    train_batch = pr.get_tensor_batch_from_json(name + ".json", batch_size, is_shuffle=is_shuffle)
    return train_batch

class InputData(object):
    # deepfashion root dir
    _data_root_dir = config_utils.get_global("deepfashion_root_dir")

    _image_dir = "Category and Attribute Prediction Benchmark/Img/"

    @staticmethod
    def get_category_chs_list():
        return json_data.get_category_chs_list()

    @staticmethod
    def get_category_list():
        return json_data.get_category_list()

    @staticmethod
    def get_list(json_name, is_shuffle=False):
        return json_data.get_list(json_name, is_shuffle)

    # get batch from json, return a tenor list [img_batch, label_batch]
    @staticmethod
    def get_tensor_batch_from_json(json_name, batch_size=32, is_shuffle=True):
        path_list, label_list, bbox_list = json_data.get_list(json_name, is_shuffle)
        datas = (path_list, label_list, bbox_list)
        if batch_size == -1:
            batch_size = len(path_list)
        batch = get_iterator(datas, batch_size=batch_size, is_shuffle=is_shuffle)
        return batch

    @staticmethod
    def get_mean(json_name):
        path_list, label_list, bbox_list = json_data.get_list(json_name)
        datas = path_list
        batch_tensor = get_iterator(datas, batch_size=1, num_epochs=1, preprocess=mean_preprocess)
        # datas = (path_list, bbox_list)
        # batch_tensor = get_iterator(datas, batch_size=1, num_epochs=1, preprocess=mean_preprocess_with_bbox)

        with tf.Session() as sess:
            all_mean = 0
            for i in range(len(path_list)):
                bc = batch_tensor.eval()
                all_mean += bc[0]
                print("step: {}, mean:{}, total mean:{}, total:{}".
                      format(i, bc[0], all_mean / (i + 1), all_mean))

            print("result: {}".format(all_mean / len(path_list)))

    def test_batch(self):
        batch_size = 16
        #中文名类别
        chs_list = self.get_category_chs_list()
        category_list = self.get_category_list()

        batch_tensor = get_data("test", batch_size=batch_size, is_shuffle=False)
        with tf.Session() as sess:
            for i in range(2500):
                img_batch, label_batch = sess.run(batch_tensor)
                for j in range(batch_size):
                    index = np.argmax(label_batch[j])
                    print(category_list[index], chs_list[index])
                    image_utils.show_image(img_batch[j].astype(np.uint8))

def main():
    pr = InputData()
    # print(pr.get_category_chs_list())
    start = time.time()
    # pr.get_mean_with_plt(json_name="prediction_train.json")
    # pr.get_mean_with_tf(json_name="prediction_train.json")
    # pr.get_json_list(json_name="prediction_train.json")
    pr.get_mean("train.json")
    print("cost time {}".format(time.time() - start))

if __name__ == '__main__':
    main()


