# encoding=utf8
# Author=LclMichael

import time

import numpy as np
import tensorflow as tf

from deepclothing.data.caltech101.json_data import JsonDataTools
from deepclothing.util import image_utils
from deepclothing.util import config_utils

train_mean = 132

train_variance = 6979.9

def image_preprocess(path, label):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.convert_image_dtype(image, dtype=tf.int64)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [224, 224])
    # image = tf.subtract(image, train_mean)
    image = tf.image.per_image_standardization(image)
    # image = tf.div(image, tf.sqrt(train_variance))
    label = tf.one_hot(label, 101)
    return image, label

def mean_preprocess(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    mean, variance = tf.nn.moments(image, [0, 1, 2])
    return mean

def variance_preprocess(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    image = tf.subtract(image, train_mean)
    result = tf.reduce_mean(tf.square(image))
    return result

class InputData(object):

    def __init__(self):
        self._data_root_dir = config_utils.get_global("caltech101_root_dir")
        self.json_data_tools = JsonDataTools()

    @staticmethod
    def get_iterator(datas, batch_size=16, threads=4, num_epochs=-1, preprocess=image_preprocess):
        dataset = tf.data.Dataset.from_tensor_slices(datas)
        dataset = dataset.map(preprocess, num_parallel_calls=threads)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        # if is_shuffle and batch_size >= 1:
        #     dataset = dataset.shuffle(batch_size * 2)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def get_mean(self, json_name="train.json"):
        datas = self.json_data_tools.get_data_list(json_name)
        batch_tensor = self.get_iterator(datas[0], batch_size=1, num_epochs=1, preprocess=mean_preprocess)
        data_len = len(datas[0])
        with tf.Session() as sess:
            all_mean = 0
            for i in range(data_len):
                bc = batch_tensor.eval()
                all_mean += bc[0]
                print("step: {}, mean:{}, total mean:{}, total:{}".
                      format(i, bc[0], all_mean / (i + 1), all_mean))
            print("result: {}".format(all_mean / data_len))

    def get_variance(self, json_name="train.json"):
        datas = self.json_data_tools.get_data_list(json_name)
        batch_tensor = self.get_iterator(datas[0], batch_size=1, num_epochs=1, preprocess=variance_preprocess)
        data_len = len(datas[0])
        with tf.Session() as sess:
            all_var = 0
            for i in range(data_len):
                bc = batch_tensor.eval()
                all_var += bc[0]
                print("step: {}, mean:{}, total mean:{}, total:{}".
                      format(i, bc[0], all_var / (i + 1), all_var))
            print("result: {}".format(all_var / data_len))

    # get batch from json, return a tenor list [img_batch, label_batch]
    def get_tensor_batch_from_json(self, json_name, batch_size=16, is_shuffle=True):
        datas = self.json_data_tools.get_data_list(json_name, is_shuffle)
        batch_size = batch_size if batch_size >= 0 else len(datas[0])
        batch = self.get_iterator(datas, batch_size=batch_size)
        return batch

    def get_category_list(self, category_file_name="category.json"):
        data = self.json_data_tools.get_category_list(category_file_name)
        return data

    def test_batch(self):
        batch_size = 30
        category_list = self.get_category_list()
        batch_tensor = get_tenosr_data("train", batch_size=batch_size, is_shuffle=False)
        with tf.Session() as sess:
            for i in range(101):
                img_batch, label_batch = sess.run(batch_tensor)
                for j in range(batch_size):
                    print(label_batch[j])
                    # print(i, category_list[category_index])
                    image_utils.show_image(np.uint8(img_batch[j]))
                    break

input_data = InputData()

def get_category_list():
    return input_data.get_category_list()

def get_tenosr_data(name="train", batch_size=16, is_shuffle=True):
    train_batch = input_data.get_tensor_batch_from_json(name + ".json", batch_size, is_shuffle=is_shuffle)
    return train_batch

def main():
    # print(pr.get_category_chs_list())
    start = time.time()
    # pr.get_mean_with_plt(json_name="prediction_train.json")
    # pr.get_mean_with_tf(json_name="prediction_train.json")
    # pr.get_json_list(json_name="prediction_train.json")
    # input_data.get_mean()
    input_data.test_batch()
    print("cost time {}".format(time.time() - start))

if __name__ == '__main__':
    main()


