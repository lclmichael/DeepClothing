# encoding=utf8
# Author=LclMichael

import os
import configparser
import codecs

import deepclothing.config as config

def get_global(name, group="global"):
    cf = configparser.ConfigParser()
    config_dir = os.path.dirname(config.__file__)
    config_path = os.path.join(config_dir, "config.ini")
    cf.read_file(codecs.open(config_path, "r", "utf-8"))
    return cf.get(group, name)


def main():
    print(get_global("deepfashion_dir"))
    pass

if __name__ == '__main__':
    main()