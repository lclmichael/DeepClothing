#encoding=utf8
#Author=LclMichael

import os
import time
import json
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = "./tfrecord/"

DEFAULT_JSON_DIR = "./json/"

image_dir = "E:/DataSet/DeepFashion/Category and Attribute Prediction Benchmark/Img/"

def read_json_file(json_path):
    with open(json_path, encoding="utf-8") as file:
        return json.load(file)

def show_image(img, num=0):
    plt.figure(num)
    plt.imshow(img)
    plt.show()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def decode_image(image_path, resize=[300, 300]):
    image = tf.gfile.FastGFile(image_path, "rb").read()
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, resize)
    return image

def build_tfrecord(output_file_path, data):
    with tf.python_io.TFRecordWriter(output_file_path) as writer, tf.Session() as sess:
        count = 0
        for json_obj in data:
            image_path = os.path.join(image_dir, json_obj["path"])
            category_num = json_obj["categoryNum"]
            image = decode_image(image_path)
            image =  image.eval()
            image_raw = image.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw":_bytes_feature(image_raw),
                        "label":_int64_feature(category_num)
            }))
            writer.write(example.SerializeToString())
            print("\r" + "success write records for " + str(count), end="")
            if count > 99:
                break

def build_all(output_dir = DEFAULT_OUTPUT_DIR):
    train_record_path = output_dir + "train.tfrecords"
    validate_record_path = output_dir + "validate.tfrecords"
    test_record_path = output_dir + "test.tfrecords"

    train_json_path = "./json/image_label_train.json"
    validate_json_path = "./json/image_label_validate.json"
    test_json_path = "./json/image_label_test.json"

    train_json = read_json_file(train_json_path)
    validate_json = read_json_file(validate_json_path)
    test_json = read_json_file(test_json_path)

    print("start write tfrecords")
    start_time = time.time()
    print("for train===>")
    build_tfrecord(train_record_path, train_json["data"])
    print("for validate===>")
    build_tfrecord(validate_record_path, validate_json["data"])
    print("for test===>")
    build_tfrecord(test_record_path, test_json["data"])
    cost_time = time.time() - start_time
    print("\n write tfrecords success, cost time: " + str(cost_time))

def set_parser():
    parser = argparse.ArgumentParser(description="this script build tfrecords data from json")
    parser.add_argument("-o", action="store", default="./tfrecord/", help="output path for file")
    parser.add_argument("-j", action="store", default="", help="base dir of json")
    parser.add_argument("-s", action="store", default="", help="base dir of deepfashion category img")

    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    FLAGS = set_parser()
    global image_dir
    if FLAGS.s != "":
        image_dir = FLAGS.s
    build_all()

if __name__ == '__main__':
    main()