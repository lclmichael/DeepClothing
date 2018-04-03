#encoding=utf8
#Author=LclMichael

import os
import time
import argparse

import tensorflow as tf

from deepclothing.util import image_utils
from deepclothing.data.prediction.input_data import PredictionReader

def get_filter(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name="filter")

def get_bias(shape):
    return tf.constant(0.1, shape=shape, name="bias")

def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name="fc_weight")

def max_pool(bottom, name):
    return tf.nn.max_pool(
        bottom,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        name=name
    )

def conv_layer(bottom, input_size, output_size, name):
    with tf.variable_scope(name):
        weight = get_filter([3, 3, input_size, output_size])
        bias = get_bias([output_size])
        convd = tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding="SAME")
        add_bias = tf.nn.bias_add(convd, bias=bias)
        relu = tf.nn.relu(add_bias)
        return relu

def fc_layer(bottom, output_size, name, keep_prob=None):
    with tf.variable_scope(name):
        flatten = tf.layers.flatten(bottom)
        weight = get_weight([flatten.get_shape().as_list()[1], output_size])
        bias = get_bias([output_size])
        fc = tf.matmul(flatten, weight)
        add_bias = tf.nn.bias_add(fc, bias=bias)
        relu = tf.nn.relu(add_bias)
        if keep_prob is None:
            return relu
        drop = tf.nn.dropout(relu, keep_prob=keep_prob)
        return drop


def get_input_data(base_dir=None, json_path ="../data/prediction/json", batch_size=32):
    pr = PredictionReader()
    pr.set_dir(base_dir, json_path)
    train_batch = pr.get_batch_from_json("prediction_train.json", batch_size)
    test_batch = pr.get_batch_from_json("prediction_val.json", batch_size)
    return train_batch, test_batch

def test_get_data():
    train_batch, test_batch = get_input_data()
    with tf.Session() as sess:
        img_batch, label_batch = sess.run(train_batch)
        for i in range(5):
            print(label_batch[i])
            image_utils.show_image(img_batch[i])

class VGG16(object):

    _output_size = 50

    def __init__(self, output_size=50):
        self._output_size = output_size
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.y_truth = tf.placeholder(dtype=tf.float32, shape=[None, self._output_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

    def get_model(self):
        conv1_1 = conv_layer(self.x, 3, 64, "conv1_1")
        conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2")
        pool1 = max_pool(conv1_2, "pool1")

        conv2_1 = conv_layer(pool1, 64, 128, "conv2_1")
        conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2")
        pool2 = max_pool(conv2_2, "pool2")

        conv3_1 = conv_layer(pool2, 128, 256, "conv3_1")
        conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2")
        conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3")
        pool3 = max_pool(conv3_3, "pool3")

        conv4_1 = conv_layer(pool3, 256, 512, "conv4_1")
        conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2")
        conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3")
        pool4 = max_pool(conv4_3, "pool4")

        conv5_1 = conv_layer(pool4, 512, 512, "conv5_1")
        conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2")
        conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3")
        pool5 = max_pool(conv5_3, "pool5")

        fc1 = fc_layer(pool5, 4096, "fc1", self.keep_prob)
        fc2 = fc_layer(fc1, 4096, "fc2", self.keep_prob)
        fc3 = fc_layer(fc2, self._output_size, "fc3")

        y = tf.nn.softmax(fc3)
        cross_entropy = -tf.reduce_mean(self.y_truth * tf.log(y))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_truth, 1))
        accuracy_step = tf.reduce_mean(tf.cast(prediction, tf.float32))
        return train_step, accuracy_step

    def train(self, train_batch_tenosr, max_iter=10000):
        train_step, accuracy_step = self.get_model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(max_iter):
                start = time.time()
                train_batch = sess.run(train_batch_tenosr)
                train_step.run(
                    feed_dict={self.x:train_batch[0], self.y_truth: train_batch[1], self.keep_prob: 0.5})
                cost = time.time() - start
                print("train step %d, cost %g" % (i, cost))
                if i % 100 == 0:
                    accuracy = accuracy_step.eval(
                        feed_dict={self.x: train_batch[0], self.y_truth: train_batch[1], self.keep_prob:1})
                    print("train step %d training accuracy: %g" % (i, accuracy))


def set_parser():
    parser = argparse.ArgumentParser(description="run test vgg16 model")
    parser.add_argument("-output", action="store", default="", help="output path for file")
    parser.add_argument("-json", action="store", default="", help="base dir of json")

    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # test_get_data()
    train_batch, test_batch = get_input_data(batch_size=32)
    vgg = VGG16()
    vgg.train(train_batch)
    pass

if __name__ == '__main__':
    main()





