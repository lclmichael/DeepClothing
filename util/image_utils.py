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

def get_image_size(file_path):
    image = Image.open(file_path)
    size = image.size
    return [size[1], size[0]]

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


def get_images_mean(paths):
    all_mean = [0, 0, 0]
    for path in paths:
        image = read_from_file(path)
        all_mean += np.mean(image, axis=(0,1))
        print("\r" + "all mean: {}".format(all_mean), end="")
    print("")
    return all_mean / len(paths)

def process_image(path, new_size, mean, is_crop=True):
    img = read_from_file(path)
    if is_crop:
        width = img.shape[1]
        heigth = img.shape[0]
        min_edge = min(width,heigth)
        xx = int((width - min_edge) / 2.0)
        yy = int((heigth - min_edge) / 2.0)
        img = img[yy : yy + min_edge, xx: xx + min_edge]
    img = np.subtract(img, mean)
    img = resize_image(img, new_size)
    img = np.array(img, dtype=np.float32)

    return img

def process_images(paths, new_size, mean, is_crop=True):
    image_list = []
    for path in paths:
        image_list.append(process_image(path, new_size, mean, is_crop))

    return image_list


