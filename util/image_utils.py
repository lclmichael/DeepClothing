#encoding=utf8
#Author=LclMichael

import matplotlib.pyplot as plt

def show_image(img, num=0):
    plt.figure(num)
    plt.imshow(img)
    plt.show()

def show_images(img_list):
    for index, img in enumerate(img_list):
        show_image(img, index)