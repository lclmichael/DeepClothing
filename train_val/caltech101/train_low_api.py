# encoding=utf8
# Author=LclMichael

import time
import argparse

import tensorflow as tf

import deepclothing.data.caltech101.input_data as input_data
import deepclothing.util.image_utils as image_utils
from deepclothing.model.low_api_vgg16 import LowApiVGG16

output_size = 101

val_data_len = 868

IMAGE_SIZE = 224

def train(lr=0.001,
          stddev=0.001,
          max_iter=200000,
          train_batch_size=16,
          val_batch_size=1,
          print_interval = 10,
          val_interval = 2000):

    saver_name = "./saver/low.ckpt"

    model = LowApiVGG16(output_size=output_size, lr=lr, stddev=stddev, image_size=IMAGE_SIZE)
    train_step_tensor = model.train_step
    loss_tensor = model.loss
    accuracy_tensor = model.accuracy

    train_data_tensor = input_data.get_tenosr_data("train", batch_size=train_batch_size, is_shuffle=True)
    val_data_tensor = input_data.get_tenosr_data("val", batch_size=val_batch_size, is_shuffle=False)

    val_iter = int(val_data_len // val_batch_size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
        very_beginning = time.time()
        start_time = very_beginning
        for i in range(max_iter):
            train_paths, train_labels = sess.run(train_data_tensor)
            train_images = image_utils.process_images(train_paths, (IMAGE_SIZE, IMAGE_SIZE), input_data.rgb_mean)
            sess.run(train_step_tensor,
                     feed_dict={model.x: train_images,
                                model.y_truth: train_labels,
                                model.is_train: True})

            if i % print_interval == 0 and i > 0:
                loss, acc = sess.run([loss_tensor, accuracy_tensor],
                                     feed_dict={model.x: train_images,
                                                model.y_truth: train_labels,
                                                model.is_train: False})
                cost_time = time.time() - start_time
                print_result("train", i, loss, acc, cost_time)
                start_time = time.time()

            if i % val_interval == 0 and i > 0:
                print("start to save: " + saver_name)
                saver.save(sess, saver_name)
                print("save success")
                total_loss = 0
                total_acc = 0
                for j in range(val_iter):
                    val_paths, val_labels = sess.run(val_data_tensor)
                    val_images = image_utils.process_images(val_paths, (IMAGE_SIZE, IMAGE_SIZE), input_data.rgb_mean)
                    loss, acc = sess.run([loss_tensor, accuracy_tensor],
                                         feed_dict={model.x: val_images,
                                                    model.y_truth: val_labels,
                                                    model.is_train: False})
                    total_loss += loss
                    total_acc += acc
                cost_time = time.time() - start_time
                # print(val_iter, total_loss, total_acc)
                print_result("val", i, total_loss / val_iter, total_acc / val_iter, cost_time)
                start_time = time.time()

        cost_time = time.time() - very_beginning
        print("train stop cost time {:.2f}".format(cost_time))


def print_result(name, step, loss, acc, cost_time, remark=""):
    print(name + " on step {}; loss: {:.5f}; accuracy:{:.3f}; cost time {:.2f}; remark:{}"
          .format(step, loss, acc, cost_time, remark))

def set_parser():
    parser = argparse.ArgumentParser(description="run train low api vgg16 network for clatech101")
    parser.add_argument("-train_batch_size", action="store", default=32, type=int, help="train batch size")
    parser.add_argument("-val_batch_size", action="store", default=1, type=int, help="val batch size")
    parser.add_argument("-lr", action="store", default=1e-3, type=float, help="learning rate")
    parser.add_argument("-stddev", action="store", default=1e-3, type=float, help="weight stddev")
    parser.add_argument("-iter", action="store", default=200000, type=int, help="max iter")
    parser.add_argument("-print_interval", action="store", default=10, type=int, help="print interval")
    parser.add_argument("-val_interval", action="store", default=2000, type=int, help="val interval")
    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

def main():
    FLAGS = set_parser()
    lr = FLAGS.lr
    stddev = FLAGS.stddev
    max_iter = FLAGS.iter
    print_interval = FLAGS.print_interval
    val_interval = FLAGS.val_interval
    train_batch_size = FLAGS.train_batch_size
    val_batch_size = FLAGS.val_batch_size
    train(lr=lr,
          stddev=stddev,
          max_iter=max_iter,
          train_batch_size=train_batch_size,
          val_batch_size=val_batch_size,
          print_interval=print_interval,
          val_interval=val_interval)
    pass
    
if __name__ == "__main__":
    main()