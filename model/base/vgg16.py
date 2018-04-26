# encoding=utf8
# Author=LclMichael

import tensorflow as tf

def conv_layer(tensor, filters, kernel_size=(3,3), name="conv_layer"):
        return tf.layers.conv2d(inputs=tensor,
                                filters=filters,
                                kernel_size=kernel_size,
                                padding="SAME",
                                activation=tf.nn.relu,
                                name=name)

def max_pool(tensor, name="pool_layer"):
    return tf.layers.max_pooling2d(tensor,
                                   pool_size=[2,2],
                                   strides=2,
                                   name=name)


def dense_layer(tensor, units, name="dense_layer"):
    return tf.layers.dense(tensor,
                           units=units,
                           activation=tf.nn.relu,
                           name=name)

def dropout_layer(tensor, rate=0.5, is_train=False, name="droput_layer"):
    return tf.layers.dropout(tensor,
                      rate=rate,
                      training=is_train,
                      name=name)

class VGG16(object):

    @staticmethod
    def get_model(input_x_tensor, is_train_tensor):
        conv1_1 = conv_layer(input_x_tensor, 64, name="conv1_1")
        conv1_2 = conv_layer(conv1_1, 64, name="conv1_2")
        pool1 = max_pool(conv1_2, name="pool1")

        conv2_1 = conv_layer(pool1, 128, name="conv2_1")
        conv2_2 = conv_layer(conv2_1, 128,  name="conv2_2")
        pool2 = max_pool(conv2_2, name="pool2")

        conv3_1 = conv_layer(pool2, 256, name="conv3_1")
        conv3_2 = conv_layer(conv3_1, 256, name="conv3_2")
        conv3_3 = conv_layer(conv3_2, 256, name="conv3_3")
        pool3 = max_pool(conv3_3, name="pool3")

        conv4_1 = conv_layer(pool3, 512, name="conv4_1")
        conv4_2 = conv_layer(conv4_1, 512, name="conv4_2")
        conv4_3 = conv_layer(conv4_2, 512, name="conv4_3")
        pool4 = max_pool(conv4_3, name="pool4")

        conv5_1 = conv_layer(pool4, 512, name="conv5_1")
        conv5_2 = conv_layer(conv5_1, 512, name="conv5_2")
        conv5_3 = conv_layer(conv5_2, 512, name="conv5_3")
        pool5 = max_pool(conv5_3, name="pool5")

        dense1 = dense_layer(pool5, units=4096, name="dense_1")
        drop1 = dropout_layer(dense1, is_train=is_train_tensor, name="drop1")
        dense2 = dense_layer(drop1, units=4096, name="dense_2")
        drop2 = dropout_layer(dense2, is_train=is_train_tensor, name="drop2")

        return drop2

def main():

    pass
    
if __name__ == "__main__":
    main()