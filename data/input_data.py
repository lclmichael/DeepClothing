# encoding=utf8
# Author=LclMichael

import tensorflow as tf

import deepclothing.util.json_utils as ju

class DeepFashionReader(object):

    _base_dir = "E:/DataSet/DeepFashion/"

    _image_dir = "Category and Attribute Prediction Benchmark/Img/"

    _json_dir = "./json/"

    def set_base_dir(self, dir_name):
        self._base_dir = dir_name

    def read_from_json(self, json_dir = _json_dir):
        pass


