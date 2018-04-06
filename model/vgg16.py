#encoding=utf8
#Author=LclMichael

import os
import time
import argparse

import tensorflow as tf

from deepclothing.util import image_utils
from deepclothing.data.prediction.input_data import PredictionReader

# filter, fully-connector layer weight
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
        name=name
    )

def conv_layer(bottom, input_size, output_size, is_train, stddev=1e-2, name="conv_layer"):
    with tf.variable_scope(name):
        weight = get_weight([3, 3, input_size, output_size], stddev=stddev, name="filter")
        bias = get_bias([output_size])
        convd = tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding="SAME")
        add_bias = tf.nn.bias_add(convd, bias=bias)
        bn = tf.layers.batch_normalization(add_bias, training=is_train)
        relu = tf.nn.relu(bn)
        return relu

def fc_layer(bottom, output_size, is_hidden, is_train, stddev=1e-2, name="fc_layer"):
    with tf.variable_scope(name):
        flatten = tf.layers.flatten(bottom)
        weight = get_weight([flatten.get_shape().as_list()[1], output_size], stddev=stddev, name="weight")
        bias = get_bias([output_size])
        fc = tf.matmul(flatten, weight)
        add_bias = tf.nn.bias_add(fc, bias=bias)
        if is_hidden:
            bn = tf.layers.batch_normalization(add_bias, training=is_train)
            relu = tf.nn.relu(bn)
            return relu
        return add_bias

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
        self.is_train = tf.placeholder(dtype=tf.bool, name="is_train")

    def get_model(self, lr=1e-3, stddev=1e-2):
        conv1_1 = conv_layer(self.x, 3, 64, self.is_train, stddev=stddev, name="conv1_1")
        conv1_2 = conv_layer(conv1_1, 64, 64, self.is_train, name="conv1_2")
        pool1 = max_pool(conv1_2, "pool1")

        conv2_1 = conv_layer(pool1, 64, 128, self.is_train, stddev=stddev, name="conv2_1")
        conv2_2 = conv_layer(conv2_1, 128, 128, self.is_train, stddev=stddev, name="conv2_2")
        pool2 = max_pool(conv2_2, "pool2")

        conv3_1 = conv_layer(pool2, 128, 256, self.is_train, stddev=stddev, name="conv3_1")
        conv3_2 = conv_layer(conv3_1, 256, 256, self.is_train, stddev=stddev, name="conv3_2")
        conv3_3 = conv_layer(conv3_2, 256, 256, self.is_train, stddev=stddev, name="conv3_3")
        pool3 = max_pool(conv3_3, "pool3")

        conv4_1 = conv_layer(pool3, 256, 512, self.is_train, stddev=stddev, name="conv4_1")
        conv4_2 = conv_layer(conv4_1, 512, 512, self.is_train, stddev=stddev, name="conv4_2")
        conv4_3 = conv_layer(conv4_2, 512, 512, self.is_train, stddev=stddev, name="conv4_3")
        pool4 = max_pool(conv4_3, "pool4")

        conv5_1 = conv_layer(pool4, 512, 512, self.is_train, stddev=stddev, name="conv5_1")
        conv5_2 = conv_layer(conv5_1, 512, 512, self.is_train, stddev=stddev, name="conv5_2")
        conv5_3 = conv_layer(conv5_2, 512, 512, self.is_train, stddev=stddev, name="conv5_3")
        pool5 = max_pool(conv5_3, "pool5")

        fc1 = fc_layer(pool5, 4096, is_hidden=True, is_train=self.is_train, stddev=stddev, name="fc1")
        fc2 = fc_layer(fc1, 4096, is_hidden=True, is_train=self.is_train, stddev=stddev, name="fc2")
        fc3 = fc_layer(fc2, self._output_size, is_hidden=False, is_train=self.is_train, stddev=stddev, name="fc3")
        y = tf.nn.softmax(fc3)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_truth, logits=fc3)
        loss = tf.reduce_mean(cross_entropy)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_truth, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        return train_step, loss, accuracy

    def train(self,
              train_batch_tenosr,
              val_batch_tensor,
              lr=1e-3,
              stddev=1e-2,
              max_iter=200000,
              print_interval=100,
              val_interval=2000):
        """

        :param train_batch_tenosr:
        :param val_batch_tensor:
        :param lr:
        :param stddev:
        :param max_iter:
        :param print_interval:
        :param val_interval:
        :return:
        """
        train_step_tensor, loss_tensor, accuracy_tensor = self.get_model(lr=lr, stddev=stddev)
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
                        self.keep_prob: 0.5,
                        self.is_train: True
                    })
                if i % print_interval == 0 and i > 0:
                    loss = sess.run(
                        loss_tensor,
                        feed_dict={
                            self.x: train_batch[0],
                            self.y_truth: train_batch[1],
                            self.keep_prob:1.0,
                            self.is_train:True
                        })
                    cost_time = time.time() - start_time
                    print("train on step {}; loss: {:.5f}; cost time {:.2f};".format(i, loss, cost_time))
                    start_time = time.time()
                if i % val_interval == 0 and i > 0:
                    start_time = time.time()
                    all_loss = 0
                    all_accuracy = 0
                    for j in range(1000):
                        val_batch = sess.run(val_batch_tensor)
                        loss, accuracy = sess.run(
                            [loss_tensor, accuracy_tensor],
                            feed_dict={
                                self.x: val_batch[0],
                                self.y_truth: val_batch[1],
                                self.keep_prob: 1.0,
                                self.is_train: False
                            })
                        all_loss += loss
                        all_accuracy += accuracy
                    cost_time = time.time() - start_time
                    print("test on step {} ; loss: {:.5f}; accuracy: {:.3f} cost time {:.2f};"
                          .format(i, all_loss / 1000, all_accuracy /1000, cost_time))
                    start_time = time.time()

def set_parser():
    parser = argparse.ArgumentParser(description="run test vgg16 model")
    parser.add_argument("-train_batch_size", action="store", default=16, type=int, help="train batch size")
    parser.add_argument("-val_batch_size", action="store", default=16, type=int, help="val batch size")
    parser.add_argument("-lr", action="store", default=1e-3, type=float, help="learning rate")
    parser.add_argument("-stddev", action="store", default=1e-3, type=float, help="weight stddev")
    parser.add_argument("-iter", action="store", default=200000, type=int, help="max iter")
    parser.add_argument("-print_interval", action="store", default=100, type=int, help="print interval")
    parser.add_argument("-val_interval", action="store", default=2000, type=int, help="val interval")
    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # test_get_data()
    FLAGS = set_parser()
    lr = FLAGS.lr
    stddev = FLAGS.stddev
    max_iter = FLAGS.iter
    print_interval = FLAGS.print_interval
    val_interval = FLAGS.val_interval
    train_batch_size = FLAGS.train_batch_size
    val_batch_size = FLAGS.val_batch_size
    train_batch = get_train_data(batch_size=train_batch_size)
    val_batch = get_val_data(batch_size=val_batch_size)
    vgg = VGG16()
    vgg.train(train_batch,
              val_batch,
              lr=lr,
              stddev=stddev,
              max_iter=max_iter,
              print_interval=print_interval,
              val_interval=val_interval)
    pass

if __name__ == '__main__':
    main()





