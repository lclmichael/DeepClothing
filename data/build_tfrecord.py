#encoding=utf8
#Author=LclMichael

import os
import time
import argparse

import numpy as np
import tensorflow as tf

import deepclothing.util.json_utils as ju
import deepclothing.util.image_utils as iu

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def decode_image(image_path, resize=[300, 300]):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, resize)
    return image

# shuffle data and 90% for train, 5% for validate, 5% for test
def shuffle_list(data):
    np.random.shuffle(data)
    count = len(data)
    train_count = int(count * 0.9)
    validate_count = int(count * 0.05)
    train_data = data[0 : train_count]
    validate_data = data[train_count : train_count + validate_count]
    test_data = data[train_count + validate_count : ]
    return train_data, validate_data, test_data

class TFRecordConverter:
    # deepfashion root dir
    _base_dir = "E:/DataSet/DeepFashion/"
    # img dir
    _image_dir = "Category and Attribute Prediction Benchmark/Img/"
    _tfrecord_dir = "./tfrecord/"
    _json_dir = "./json/"

    def set_base_dir(self, dir_name):
        self._base_dir = dir_name

    def set_tfrecord_dir(self, dir_name):
        self._tfrecord_dir = dir_name

    def set_json_dir(self, dir_name):
        self._json_dir = dir_name

    def build_tfrecord(self, output_file_path, data, batch_size = 1, num_threads=8):
        image_list = []
        label_list = []
        json_count = 0

        for json_obj in data:
            json_count += 1
            image_list.append(os.path.join(self._base_dir, self._image_dir, json_obj["path"]))
            label_list.append(json_obj["categoryNum"])
            if json_count >= 100:
                break

        image_queue, label_queue = tf.train.slice_input_producer([image_list, label_list], num_epochs=1)
        image_batch, label_batch = tf.train.shuffle_batch(
            [decode_image(image_queue), label_queue],
            batch_size=batch_size,
            capacity=batch_size * 10,
            min_after_dequeue=batch_size,
            num_threads=num_threads,
            shapes=[(300, 300, 3), ()])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess, tf.python_io.TFRecordWriter(output_file_path) as writer:
            coord = tf.train.Coordinator()
            sess.run(tf.local_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                count = 0
                while not coord.should_stop():
                    image_batch_run, label_batch_run = sess.run([image_batch, label_batch])
                    for i in range(batch_size):
                        image_raw = image_batch_run[i].tostring()
                        example = tf.train.Example(
                             features=tf.train.Features(
                                 feature={
                                     "image_raw":_bytes_feature(image_raw),
                                     "label": _int64_feature(label_batch_run[i])
                                 }
                             )
                        )
                        writer.write(example.SerializeToString())
                        count += 1
                        print("\r" + "write records for " + str(count), end="")
                print("\n<======= end")
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()

    def build_all(self):
        train_record_path = os.path.join(self._tfrecord_dir, "train.tfrecords")
        validate_record_path = os.path.join(self._tfrecord_dir, "validate.tfrecords")
        test_record_path = os.path.join(self._tfrecord_dir, "test.tfrecords")

        label_json_path = os.path.join(self._json_dir, "image_label.json")

        label_json = ju.read_json_file(label_json_path)
        train_data, validate_data, test_data = shuffle_list(label_json["data"])

        print("start write tfrecords")
        start_time = time.time()
        print("======> for train count : " + str(len(train_data)))
        self.build_tfrecord(train_record_path, train_data)

        print("======> for validate count : " + str(len(validate_data)))
        self.build_tfrecord(validate_record_path, validate_data)

        print("======> for test count : " + str(len(test_data)))
        self.build_tfrecord(test_record_path, test_data)
        cost_time = time.time() - start_time
        print("write tfrecords success, cost time: " + str(cost_time))

def set_parser():

    parser = argparse.ArgumentParser(description="this script build tfrecords data from json")
    parser.add_argument("-output", action="store", default="", help="output path for file")
    parser.add_argument("-json", action="store", default="", help="base dir of json")
    parser.add_argument("-deepfashion", action="store", default="", help="base dir of deepfashion category img")

    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    FLAGS = set_parser()
    tc = TFRecordConverter()
    output_dir = FLAGS.output
    json_dir = FLAGS.json
    deepfashion_dir = FLAGS.deepfashion
    if output_dir != "":
        tc.set_tfrecord_dir(output_dir)
    if json_dir != "":
        tc.set_json_dir(json_dir)
    if deepfashion_dir != "":
        tc.set_base_dir(deepfashion_dir)
    tc.build_all()

if __name__ == '__main__':
    main()