# encoding=utf8
# Author=LclMichael

import os
import argparse

from deepclothing.util import json_utils
from deepclothing.util import config_utils

class CovertToJson(object):
    # deepfashion root dir
    _data_root_dir = config_utils.get_global("deepfashion_root_dir")
    # deepfashion category anno 标注文件目录
    _anno_dir = "Category and Attribute Prediction Benchmark/Anno/"
    # 划分训练，测试还有泛化数据集
    _eval_dir = "Category and Attribute Prediction Benchmark/Eval/"

    #类别信息标注文件
    _list_category_cloth_path = os.path.join(_anno_dir, "list_category_cloth.txt")
    #中文标注文件
    _list_category_cloth_chn_path = os.path.join(_anno_dir, "list_category_cloth_chs.txt")
    #图片分类标注文件
    _list_category_image_path = os.path.join(_anno_dir, "list_category_img.txt")
    #候选框标注文件
    _list_bbox_path = os.path.join(_anno_dir, "list_bbox.txt")

    _list_eval_path = os.path.join(_eval_dir, "list_eval_partition.txt")

    _json_dir = "./json/"

    def set_dir(self, dir_name):
        self._json_dir = dir_name

    def get_category_chn_list(self):
        file_path = os.path.join(self._data_root_dir, self._list_category_cloth_chn_path)
        with open(file_path, "r", encoding="utf-8") as file:
            datas = file.readlines()
            category_list = [line.strip().split()[0] for line in datas]
        return category_list

    # 从deepfashion的list_category_cloth.txt中获取类别数组
    def get_category_list(self):
        file_path = os.path.join(self._data_root_dir, self._list_category_cloth_path)
        with open(file_path, "r") as file:
            datas = file.readlines()[2:]
            category_list = [line.strip().split()[0] for line in datas]
        return category_list

    def get_eval_dict(self):
        file_path = os.path.join(self._data_root_dir, self._list_eval_path)
        with open(file_path, "r") as file:
            datas = file.readlines()[2:]
            eval_dict = {line.strip().split()[0]: line.strip().split()[1] for line in datas}
        return eval_dict

    # 格式[{categoryNum:1,bbox:[x0,y0,width,height],path:"xxx/yyy/zz.jpg"},...]
    def get_all_list(self):
        bbox_file_path = os.path.join(self._data_root_dir, self._list_bbox_path)
        image_path = os.path.join(self._data_root_dir, self._list_category_image_path)
        with open(bbox_file_path, "r") as bb_image_file, open(image_path, "r") as image_file:
            bbox_image_list = [line.split() for line in bb_image_file.readlines()][2:]
            category_image_list = [line.split() for line in image_file.readlines()][2:]
            all_list = []
            category_label_dict  = {}
            for category_image in category_image_list:
                path=category_image[0]
                category_label_dict[path] = int(category_image[1]) - 1

            for index, bbox_image in enumerate(bbox_image_list):
                path = bbox_image[0]
                bbox = [int(x) for x in bbox_image[1:]]
                #img/Striped_A-Line_Dress/img_00000003.jpg w150 h225 [84, 40, 184, 211] 实际:x2应为120
                #img/Striped_A-Line_Dress/img_00000006.jpg w300 h273 [1, 1, 200, 300] 实际:[74, 44, 157, 187]
                #img/Striped_A-Line_Dress/img_00000009.jpg w202 h300 [5, 30, 209, 300] 实际:[45, 37, 144, 205]
                #img/Striped_A-Line_Dress/img_00000010.jpg w200 h300 [1, 1, 225, 299] 实际:[35, 14, 181, 245]
                if path == "img/Striped_A-Line_Dress/img_00000003.jpg":
                    bbox[2] = 120
                elif path == "img/Striped_A-Line_Dress/img_00000006.jpg":
                    bbox = [74, 44, 157, 187]
                elif path == "img/Striped_A-Line_Dress/img_00000009.jpg":
                    bbox = [45, 37, 144, 205]
                elif path == "img/Striped_A-Line_Dress/img_00000010.jpg":
                    bbox = [35, 14, 181, 245]
                jsonObj = {"id":index, "path": path, "bbox": bbox, "categoryNum": category_label_dict[path]}
                all_list.append(jsonObj)

        return all_list

    def get_partition_list(self, all_list):
        eval_dict = self.get_eval_dict()
        train_list = []
        val_list = []
        test_list = []
        for obj in all_list:
            if eval_dict[obj["path"]] == "train":
                train_list.append(obj)
            elif eval_dict[obj["path"]] == "val":
                val_list.append(obj)
            elif eval_dict[obj["path"]] == "test":
                test_list.append(obj)

        return train_list, val_list, test_list

    def build_all_json_file(self):
        category_list = self.get_category_list()
        chn_list = self.get_category_chn_list()
        all_list = self.get_all_list()
        train_list, val_list, test_list = self.get_partition_list(all_list)
        json_utils.write_json_file(chn_list, self._json_dir, "prediction_category_chs.json")
        json_utils.write_json_file(category_list, self._json_dir, "prediction_category.json")
        json_utils.write_json_file(all_list, self._json_dir, "prediction_all.json")
        json_utils.write_json_file(train_list, self._json_dir, "prediction_train.json")
        json_utils.write_json_file(val_list, self._json_dir, "prediction_val.json")
        json_utils.write_json_file(test_list, self._json_dir, "prediction_test.json")

def set_parser():
    parser = argparse.ArgumentParser(description="this script build json data from deepfashion")
    parser.add_argument("-output", action="store", default="", help="output path for json file")
    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

# 两个参数一个-output, 表示输出目录, 一个-source 表示deepfashion目录
# 脚本用于生成json文件,包括属性列表json,所有图片数据json
def main():
    FLAGS = set_parser()
    dc = CovertToJson()
    json_dir = FLAGS.output
    if json_dir != "":
        dc.set_dir(json_dir)
    dc.build_all_json_file()
    # dc.get_all_list()
if __name__ == '__main__':
    main()
