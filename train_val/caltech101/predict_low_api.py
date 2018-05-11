# encoding=utf8
# Author=LclMichael

import time
import argparse

import numpy as np
import tensorflow as tf

from deepclothing.model.low_api_vgg16 import LowApiVGG16
import deepclothing.util.image_utils as image_utils

output_size = 101

val_data_len = 2945

rgb_mean = [139.09414673, 132.65591431, 124.21406555]

def predict(image_path):

    very_beginning = time.time()
    saver_name = "./saver/low.ckpt"
    img = image_utils.precess_image(image_path, (224, 224), rgb_mean)

    model = LowApiVGG16(output_size=output_size)
    prediction_tensor  = model.prediction
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=1)
        # tf.global_variables_initializer().run()
        saver.restore(sess, saver_name)
        result = sess.run(prediction_tensor, feed_dict={model.x:img, model.is_train:False})
        print(np.argmax(result, 1))
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