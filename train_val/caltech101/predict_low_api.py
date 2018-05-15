# encoding=utf8
# Author=LclMichael

import os
import time
import argparse

import numpy as np
import tensorflow as tf

import deepclothing.util.image_utils as image_utils
import deepclothing.data.caltech101.json_data as json_data
from deepclothing.model.low_api_vgg16 import LowApiVGG16


output_size = 101

IMAGE_SIZE = 224

rgb_mean = [ 135.94625854, 131.7210083,123.35960388]

def predict(image_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    saver_name = "./saver/low.ckpt"
    img = image_utils.process_image(image_path, (IMAGE_SIZE, IMAGE_SIZE), rgb_mean)

    category_list = json_data.get_category_list()
    model = LowApiVGG16(output_size=output_size, image_size=IMAGE_SIZE)
    prediction_tensor  = model.y
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, saver_name)
        very_beginning = time.time()
        result = sess.run(prediction_tensor, feed_dict={model.x:[img], model.is_train:False})
        result_index = np.argmax(result, 1)[0]

        print(result_index, result[0][result_index], category_list[result_index])
        print(result)
        cost_time = time.time() - very_beginning
        print("predict cost time {:.2f}".format(cost_time))


def set_parser():
    parser = argparse.ArgumentParser(description="run predict low api vgg16 network for clatech101")
    parser.add_argument("-f", action="store", default="", help="test file path")
    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    FLAGS = set_parser()
    test_file = FLAGS.f
    predict(test_file)
    pass
    
if __name__ == "__main__":
    main()