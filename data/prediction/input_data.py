# encoding=utf8
# Author=LclMichael

import os

import tensorflow as tf

import deepclothing.util.json_utils as ju
import deepclothing.util.image_utils as iu

def decode_original_image(image, label):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [224, 224])
    label = tf.one_hot(label, 50)
    label = tf.cast(label, dtype=tf.float32)
    return image, label

def get_iterator(tensors, batch_size=32, threads=8, num_epochs=-1, is_shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.map(decode_original_image, num_parallel_calls=threads)
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    if is_shuffle:
        dataset = dataset.shuffle(10000)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

class PredictionReader(object):

    _base_dir = "E:/DataSet/DeepFashion/"

    _image_dir = "Category and Attribute Prediction Benchmark/Img/"

    _json_dir = "./json/"

    def set_base_dir(self, dir_name):
        self._base_dir = dir_name

    def set_json_dir(self, dir_name):
        self._json_dir = dir_name

    # get batch from json, return a tenor list [img_batch, label_batch]
    def get_batch_from_json(self, json_name, batch_size=32):
        data = ju.read_json_file(os.path.join(self._json_dir, json_name))

        img_list = []
        label_list = []
        for item in data:
            img_list.append(os.path.join(self._base_dir, self._image_dir, item["path"]))
            label_list.append(item["categoryNum"])
        tensors = (img_list, label_list)
        batch = get_iterator(tensors, batch_size=batch_size)
        return batch

    def test_batch(self):
        train_json_file = "prediction_train.json"
        batch_size = 32
        batch = self.get_batch_from_json(train_json_file, batch_size)
        with tf.Session() as sess:
            img_batch, label_batch = sess.run(batch)
            for i in range(batch_size):
                print(label_batch[i])
                iu.show_image(img_batch[i])

def main():
    pr = PredictionReader()
    pr.test_batch()

if __name__ == '__main__':
    main()


