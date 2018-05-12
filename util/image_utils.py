#encoding=utf8
#Author=LclMichael

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from scipy import misc

def read_from_file(file_path):
    image = Image.open(file_path)
    image = image.convert("RGB")
    return np.array(image)

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
def crop_image(img, y1, x1, y2, x2):
    return img[y1:y2, x1:x2]

#获取单张图片均值
def get_image_mean(img):
    img = np.array(img)
    return np.mean(img.flatten())

def process_image(path, size, mean):
    img = read_from_file(path)
    img = resize_image(img, size)
    img = np.array(img, dtype=np.float32)
    img = np.subtract(img, mean)
    return img


