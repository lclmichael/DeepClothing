#encoding=utf8
#Author=LclMichael

import os
import time
import argparse

import tensorflow as tf

from deepclothing.util import image_utils
from deepclothing.data.prediction.input_data import PredictionReader

# filter, fully-connector layer weight
def get_weight(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), dtype=tf.float32, name=name)

def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name="bias"))

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
        weight = get_weight([3, 3, input_size, output_size], "filter")
        bias = get_bias([output_size])
        convd = tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding="SAME")
        add_bias = tf.nn.bias_add(convd, bias=bias)
        relu = tf.nn.relu(add_bias)
        return relu

def fc_layer(bottom, output_size, name, keep_prob=None):
    with tf.variable_scope(name):
        flatten = tf.layers.flatten(bottom)
        weight = get_weight([flatten.get_shape().as_list()[1], output_size], name="weight")
        bias = get_bias([output_size])
        fc = tf.matmul(flatten, weight)
        add_bias = tf.nn.bias_add(fc, bias=bias)
        if keep_prob is None:
            return add_bias
        relu = tf.nn.relu(add_bias)
        drop = tf.nn.dropout(relu, keep_prob=keep_prob)
        return drop

def get_train_data(data_root_dir=None, json_path ="../data/prediction/json", batch_size=32):
    pr = PredictionReader()
    pr.set_dir(data_root_dir, json_path)
    train_batch = pr.get_batch_from_json("prediction_train.json", batch_size)
    return train_batch

def get_val_data(data_root_dir=None, json_path="../data/prediction/json", batch_size=32):
    pr = PredictionReader()
    pr.set_dir(data_root_dir, json_path)
    test_batch = pr.get_batch_from_json("prediction_val.json", batch_size=batch_size, is_shuffle=False)
    return test_batch

def test_get_data():
    train_batch = get_train_data()
    val_batch = get_val_data()
    with tf.Session() as sess:
        train_img_batch, train_label_batch = sess.run(train_batch)
        val_img_batch, val_label_batch = sess.run(val_batch)
        for i in range(3):
            print(train_label_batch[i])
            image_utils.show_image(train_img_batch[i])

        for i in range(3):
            print(val_label_batch[i])
            image_utils.show_image(val_img_batch[i])

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
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_truth, logits=fc3)
        loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
        prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_truth, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        return train_step, loss, accuracy

    def train(self, train_batch_tenosr, val_batch_tensor, max_iter=200000):
        train_step_tensor, loss_tensor, accuracy_tensor = self.get_model()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            start_time = time.time()
            for i in range(max_iter):
                train_batch = sess.run(train_batch_tenosr)
                sess.run(
                    train_step_tensor,
                    feed_dict={
                        self.x: train_batch[0],
                        self.y_truth: train_batch[1],
                        self.keep_prob: 0.5
                    })
                if i % 10 == 0:
                    loss = sess.run(
                        loss_tensor,
                        feed_dict={
                            self.x: train_batch[0],
                            self.y_truth: train_batch[1],
                            self.keep_prob:1.0
                        })
                    cost_time = time.time() - start_time
                    print("train on step {} ; loss: {:.5f}; cost time {:.2f};".format(i, loss, cost_time))
                    start_time = time.time()
                if i % 100 == 0:
                    start_time = time.time()
                    all_loss = 0
                    all_accuracy = 0
                    for j in range(1000):
                        val_batch = sess.run(val_batch_tensor)
                        loss, accuracy = sess.run(
                            [loss_tensor, accuracy_tensor],
                            feed_dict={
                                self.x: train_batch[0],
                                self.y_truth: train_batch[1],
                                self.keep_prob: 1.0
                            })
                        all_loss += loss
                        all_accuracy += accuracy
                    cost_time = time.time() - start_time
                    print("test on step {} ; loss: {:.5f}; accuracy: {:.3f} cost time {:.2f};"
                          .format(i, all_loss / 1000, all_accuracy /1000, cost_time))
                    start_time = time.time()

def set_parser():
    parser = argparse.ArgumentParser(description="run test vgg16 model")
    parser.add_argument("-output", action="store", default="", help="output path for file")
    parser.add_argument("-json", action="store", default="", help="base dir of json")

    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # test_get_data()
    train_batch = get_train_data(batch_size=32)
    val_batch = get_val_data(batch_size=40)
    vgg = VGG16()
    vgg.train(train_batch, val_batch, max_iter=200000)
    pass

if __name__ == '__main__':
    main()





