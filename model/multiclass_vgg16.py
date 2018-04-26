# encoding=utf8
# Author=LclMichael

import numpy as np
import tensorflow as tf

from deepclothing.model.base.vgg16 import VGG16

#base on lighten vgg



class MultiClassNetwork(object):

    def __init__(self, output_size=101, lr=0.01, stddev=0.01):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.y_truth = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.output_size = output_size
        self.learning_rate = lr
        self.stddev = stddev
        self.lighten_vgg = VGG16()

    def dense_layer(self, input_tensor, output_size, use_bias=True):
        fc = tf.layers.dense(
            inputs=tf.layers.flatten(input_tensor),
            units=output_size,
            kernel_initializer=tf.initializers.truncated_normal(stddev=self.stddev),
            use_bias=use_bias)
        # bn = tf.layers.batch_normalization(fc, training=self.is_train)
        return fc

    def build_model(self):
        features = self.lighten_vgg.get_model(self.input_x, self.is_train)
        #直接接入全连接层输出
        # fc1 = self.dense_layer(pool5, 4096)
        # fc2 = self.dense_layer(fc1, 4096)
        logits = self.dense_layer(features, self.output_size)
        y_prediction = tf.nn.softmax(logits)
        comparison = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(self.y_truth, 1))
        accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y_truth, logits=logits)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_truth, logits=logits))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return train_step, loss, accuracy, y_prediction

def main():

    pass
    
if __name__ == "__main__":
    main()