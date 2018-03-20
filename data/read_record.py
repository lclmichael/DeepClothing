# encoding="utf8"
# Author="LclMichael"

import tensorflow as tf
import matplotlib.pyplot  as plt

RECORD_PATH = "./tfrecord/test.tfrecords"

def show_image(img, num=0):
    plt.figure(num)
    plt.imshow(img)
    plt.show()

def load_record(record_path):
    reader = tf.TFRecordReader()
    filename_queque = tf.train.string_input_producer([record_path])
    _, serialized_example = reader.read(filename_queque)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features["image_raw"], tf.float32)
    image = tf.reshape(image, [300, 300, 3])
    label = tf.cast(features["label"], tf.int64)
    with tf.Session() as sess:
        coor = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coor)
        try:
            while not coor.should_stop():
                newImage, newLabel = sess.run([image, label])
                show_image(newImage)
        except tf.errors.OutOfRangeError:
            coor.request_stop()
            print("done!")
        finally:
            coor.request_stop()

def main():
    load_record(RECORD_PATH)
    pass


if __name__ == '__main__':
    main()
