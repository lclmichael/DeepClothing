# encoding=utf8
# Author=LclMichael

import numpy as np
import tensorflow as tf

from deepclothing.model.base.lighten_vgg import LightenVGG

#base on lighten vgg
class MultiClassNetwork(object):

    def __init__(self, output_size=101, lr=0.001, stddev=0.001):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.y_truth = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.output_size = output_size
        self.learning_rate = lr
        self.stddev = stddev
        self.lighten_vgg = LightenVGG()

    def build_model(self):
        pool5 = self.lighten_vgg.get_model(self.input_x, self.is_train, self.stddev)
        #直接接入全连接层输出
        fc = tf.layers.dense(inputs=tf.layers.flatten(pool5),
                             units=self.output_size,
                             kernel_initializer=tf.initializers.truncated_normal(stddev=self.stddev),
                             use_bias=False)

        logits = tf.layers.batch_normalization(fc, training=self.is_train)
        y_prediction = tf.nn.softmax(logits)
        comparison = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(self.y_truth, 1))
        accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_truth, logits=logits))
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return train_step, loss, accuracy, y_prediction

def main():

    pass
    
if __name__ == "__main__":
    main()