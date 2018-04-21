# encoding=utf8
# Author=LclMichael

import numpy as np
import tensorflow as tf

def get_weight(shape, stddev=1e-2, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), dtype=tf.float32, name=name)

def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name="bias"))

def max_pool(bottom, name):
    return tf.nn.max_pool(
        bottom,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name=name)

def conv_layer(bottom, input_size, output_size, is_train, stddev=1e-2, name="conv_layer"):
    with tf.variable_scope(name):
        weight = get_weight([3, 3, input_size, output_size], stddev=stddev, name=name + "filter")
        convd = tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding="SAME")
        bn = tf.layers.batch_normalization(convd, training=is_train, name=name + "bn")
        relu = tf.nn.relu(bn)
        return relu

class LightenVGG(object):

    @staticmethod
    def get_model(input_x_tensor, is_train_tensor, stddev=1e-2):
        conv1_1 = conv_layer(input_x_tensor, 3, 64, is_train_tensor, stddev=stddev, name="conv1_1")
        conv1_2 = conv_layer(conv1_1, 64, 64, is_train_tensor, name="conv1_2")
        pool1 = max_pool(conv1_2, "pool1")

        conv2_1 = conv_layer(pool1, 64, 128, is_train_tensor, stddev=stddev, name="conv2_1")
        conv2_2 = conv_layer(conv2_1, 128, 128, is_train_tensor, stddev=stddev, name="conv2_2")
        pool2 = max_pool(conv2_2, "pool2")

        conv3_1 = conv_layer(pool2, 128, 256, is_train_tensor, stddev=stddev, name="conv3_1")
        conv3_2 = conv_layer(conv3_1, 256, 256, is_train_tensor, stddev=stddev, name="conv3_2")
        conv3_3 = conv_layer(conv3_2, 256, 256, is_train_tensor, stddev=stddev, name="conv3_3")
        pool3 = max_pool(conv3_3, "pool3")

        conv4_1 = conv_layer(pool3, 256, 512, is_train_tensor, stddev=stddev, name="conv4_1")
        conv4_2 = conv_layer(conv4_1, 512, 512, is_train_tensor, stddev=stddev, name="conv4_2")
        conv4_3 = conv_layer(conv4_2, 512, 512, is_train_tensor, stddev=stddev, name="conv4_3")
        pool4 = max_pool(conv4_3, "pool4")

        conv5_1 = conv_layer(pool4, 512, 512, is_train_tensor, stddev=stddev, name="conv5_1")
        conv5_2 = conv_layer(conv5_1, 512, 512, is_train_tensor, stddev=stddev, name="conv5_2")
        conv5_3 = conv_layer(conv5_2, 512, 512, is_train_tensor, stddev=stddev, name="conv5_3")
        pool5 = max_pool(conv5_3, "pool5")

        return pool5

def main():

    pass
    
if __name__ == "__main__":
    main()