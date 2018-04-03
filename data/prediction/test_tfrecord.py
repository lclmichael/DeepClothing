#encoding=utf8
#Author=LclMichael

import tensorflow as tf

from deepclothing.util import image_utils

tfrecord_path = "./tfrecord/train.tfrecords"

def main():
    queue = tf.train.string_input_producer([tfrecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features["image_raw"], tf.float32)
    image = tf.reshape(image, [300, 300, 3])
    label = tf.cast(features["label"], tf.int64)
    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess, coord=coord)
        data_image, data_label = sess.run([image, label])
        image_utils.show_image(data_image)

if __name__ == '__main__':
    main()