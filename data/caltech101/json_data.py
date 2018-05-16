# encoding=utf8
# Author=LclMichael

import os

import numpy as np

from deepclothing.util import json_utils
from deepclothing.util import config_utils
import deepclothing.util.image_utils as image_utils

class JsonDataTools(object):

    _data_root_dir = config_utils.get_global("caltech101_root_dir")

    _json_dir = os.path.join(os.path.dirname(__file__), "json")

    def build_json_file(self):
        category_list = []
        json_all_list = []
        json_train_list = []
        json_val_list = []
        train_count_list = [0 for x in range(101)]
        val_count_list = [0 for x in range(101)]
        image_index = -1
        category_index = -1
        print("prepare content:")
        for root, dirs, files in os.walk(self._data_root_dir, topdown=True):
            category_name = os.path.split(root)[1]
            if category_name == "101_ObjectCategories" or category_name == "BACKGROUND_Google":
                continue
            category_index += 1
            category_list.append(category_name)
            single_count = -1
            for file_name in files:
                image_index += 1
                single_count += 1
                file_path = os.path.join(root, file_name)
                size = image_utils.get_image_size(file_path)
                json_data = {"id":image_index,
                             "categoryNum":category_index,
                             "size":size,
                             "path":os.path.join(category_name, file_name)}
                if image_index % 10 != 0 and image_index >= 0:
                    json_train_list.append(json_data)
                    train_count_list[category_index] += 1
                else:
                    val_count_list[category_index] += 1
                    json_val_list.append(json_data)

                json_all_list.append(json_data)

                print("\r" + "done for {}".format(image_index), end="")

            # for file_name in files:
            #     image_index += 1
            #     single_count += 1
            #     json_data = {"id":image_index,
            #                  "categoryNum":category_index,
            #                  "path":os.path.join(category_name, file_name)}
            #     if train_count_list[category_index] < 30:
            #         json_train_list.append(json_data)
            #         train_count_list[category_index] += 1
            #     elif val_count_list[category_index] < 50:
            #         val_count_list[category_index] += 1
            #         json_val_list.append(json_data)
            #
            #     json_all_list.append(json_data)
        print("\n" + "count num. all:{}, train:{}, val:{}, category_num:{}".
              format(len(json_all_list), len(json_train_list), len(json_val_list), len(category_list)))

        json_utils.write_json_file(category_list, self._json_dir, "category.json")
        json_utils.write_json_file(json_all_list, self._json_dir, "all.json")
        json_utils.write_json_file(json_train_list, self._json_dir, "train.json")
        json_utils.write_json_file(json_val_list, self._json_dir, "val.json")

    def get_json(self, json_name):
        return json_utils.read_json_file(os.path.join(self._json_dir, json_name))

    def get_data_list(self, file_name, is_shuffle = False):
        json_data = self.get_json(file_name)
        if is_shuffle:
            np.random.shuffle(json_data)
        path_list = []
        label_list = []
        for _data in json_data:
            path_list.append(os.path.join(self._data_root_dir, _data["path"]))
            label_list.append(_data["categoryNum"])
        return path_list, label_list

    def get_category_list(self, json_name):
        return self.get_json(json_name)

json_data_tools = JsonDataTools()

def get_json(json_name):
    return json_data_tools.get_json(json_name)

def get_category_list(json_name="category.json"):
    return json_data_tools.get_category_list(json_name)

#计算每个类别图像数量
def calc_all(json_name):
    data = get_json(json_name)
    count_list = [0 for i in range(101)]
    for obj in data:
        count_list[obj["categoryNum"]] += 1
    print(np.sum(count_list))
    print(count_list)

def main():
    jdt = JsonDataTools()
    jdt.build_json_file()
    pass
    
if __name__ == "__main__":
    main()