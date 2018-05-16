# encoding=utf8
# Author=LclMichael

import tensorflow as tf

from deepclothing.data.caltech101.json_data import json_data_tools

import deepclothing.util.image_utils as image_utils

def get_mean(file = "train.json"):
    datas = json_data_tools.get_data_list(file)
    print(image_utils.get_images_mean(datas[0]))

def main():
    get_mean()
    pass
    
if __name__ == "__main__":
    main()