#encoding=utf8
#Author=LclMichael

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

from scipy import misc

def read_from_file(file_path):
    return mpimage.imread(file_path)

def show_image(img, num=0):
    plt.figure(num)
    plt.imshow(img)
    plt.show()

def show_images(img_list):
    for index, img in enumerate(img_list):
        show_image(img, index)

#重新resize图像大小，size为[height, width]
def resize_image(img, size):
    return misc.imresize(img, size)

#裁剪图片
def crop_image(img, start_y, start_x, height, width):
    return img[start_y:start_y+height, start_x:start_x+width]

#获取单张图片均值
def get_image_mean(img):
    img = np.array(img)
    return np.mean(img.flatten())

