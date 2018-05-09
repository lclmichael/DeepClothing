# encoding=utf8
# Author=LclMichael

import os
import time
import argparse

import tensorflow as tf


# filter, fully-connector layer weight
def get_weight(shape, stddev=1e-2, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), dtype=tf.float32, name=name)

def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name="bias"))

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME",
                          name=name)


def conv_layer(bottom, input_size, output_size, is_train, stddev=1e-2, name="conv_layer"):
    with tf.variable_scope(name):
        weight = get_weight([3, 3, input_size, output_size], stddev=stddev, name="filter")
        convd = tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding="SAME")
        # add_bias = tf.nn.bias_add(convd, bias=bias)
        bn = tf.layers.batch_normalization(convd, training=is_train)
        relu = tf.nn.relu(bn)
        return relu

def fc_layer(bottom, output_size, is_hidden, is_train, stddev=1e-2, name="fc_layer"):
    with tf.variable_scope(name):
        flatten = tf.layers.flatten(bottom)
        weight = get_weight([flatten.get_shape().as_list()[1], output_size], stddev=stddev, name="weight")
        bias = get_bias([output_size])
        fc = tf.matmul(flatten, weight)
        bn = tf.layers.batch_normalization(fc, training=is_train)
        if is_hidden:

            relu = tf.nn.relu(bn)
            return relu
        add_bias = tf.nn.bias_add(fc, bias=bias)
        return add_bias

class LowApiVGG16(object):

    def __init__(self, output_size=50, lr=1e-2, stddev=1e-2):
        self.lr = lr
        self.stddev = stddev
        self._output_size = output_size
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.y_truth = tf.placeholder(dtype=tf.float32, shape=[None, self._output_size])
        self.is_train = tf.placeholder(dtype=tf.bool, name="is_train")

        conv1_1 = conv_layer(self.x, 3, 64, self.is_train, stddev=self.stddev, name="conv1_1")
        conv1_2 = conv_layer(conv1_1, 64, 64, self.is_train, name="conv1_2")
        pool1 = max_pool(conv1_2, "pool1")

        conv2_1 = conv_layer(pool1, 64, 128, self.is_train, stddev=self.stddev, name="conv2_1")
        conv2_2 = conv_layer(conv2_1, 128, 128, self.is_train, stddev=self.stddev, name="conv2_2")
        pool2 = max_pool(conv2_2, "pool2")

        conv3_1 = conv_layer(pool2, 128, 256, self.is_train, stddev=self.stddev, name="conv3_1")
        conv3_2 = conv_layer(conv3_1, 256, 256, self.is_train, stddev=self.stddev, name="conv3_2")
        conv3_3 = conv_layer(conv3_2, 256, 256, self.is_train, stddev=self.stddev, name="conv3_3")
        pool3 = max_pool(conv3_3, "pool3")

        conv4_1 = conv_layer(pool3, 256, 512, self.is_train, stddev=self.stddev, name="conv4_1")
        conv4_2 = conv_layer(conv4_1, 512, 512, self.is_train, stddev=self.stddev, name="conv4_2")
        conv4_3 = conv_layer(conv4_2, 512, 512, self.is_train, stddev=self.stddev, name="conv4_3")
        pool4 = max_pool(conv4_3, "pool4")

        conv5_1 = conv_layer(pool4, 512, 512, self.is_train, stddev=self.stddev, name="conv5_1")
        conv5_2 = conv_layer(conv5_1, 512, 512, self.is_train, stddev=self.stddev, name="conv5_2")
        conv5_3 = conv_layer(conv5_2, 512, 512, self.is_train, stddev=self.stddev, name="conv5_3")
        pool5 = max_pool(conv5_3, "pool5")

        # fc1 = fc_layer(pool5, 4096, is_hidden=True, is_train=self.is_train, self.stddev=self.stddev, name="fc1")
        # fc2 = fc_layer(fc1, 4096, is_hidden=True, is_train=self.is_train, self.stddev=self.stddev, name="fc2")
        # fc3 = fc_layer(fc2, self._output_size, is_hidden=False, is_train=self.is_train, self.stddev=self.stddev, name="fc3")
        fc = fc_layer(pool5, self._output_size, is_hidden=False, is_train=self.is_train, stddev=self.stddev, name="fc")
        self.y = tf.nn.softmax(fc)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_truth, logits=fc)
        self.loss = tf.reduce_mean(cross_entropy)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_truth, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

def main():

    pass
    
if __name__ == "__main__":
    main()