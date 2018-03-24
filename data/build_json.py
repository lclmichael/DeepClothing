# encoding=utf8
# Author=LclMichael

import argparse

import deepclothing.util.json_utils as ju


class DeepFashionConverter:

    # deepfashion root dir
    _base_dir = "E:/DataSet/DeepFashion/"
    # deepfashion category anno 标注文件目录
    _category_dir = "Category and Attribute Prediction Benchmark/Anno/"
    #类别信息标注文件
    _list_category_cloth_path = _category_dir + "list_category_cloth.txt"
    #图片分类标注文件
    _list_category_image_path = _category_dir + "list_category_img.txt"
    #候选框标注文件
    _list_bbox_path = _category_dir + "list_bbox.txt"
    _json_dir = "./json/"

    def set__base_dir(self, dir_name):
        self._base_dir = dir_name

    def set__json_dir(self, dir_name):
        self._json_dir = dir_name

    # 从deepfashion的list_category_cloth.txt中获取类别字典
    def get_category_dict(self):
        file_path = self._base_dir + self._list_category_cloth_path
        with open(file_path, "r") as file:
            datas = file.readlines()[2:]
            category_dict = {line.strip().split()[0]: index for index, line in enumerate(datas)}
            file.close()
        return category_dict

    # 从deepfashion的list_category_cloth.txt中获取类别数组
    def get_category_list(self):
        file_path = self._base_dir + self._list_category_cloth_path
        with open(file_path, "r") as file:
            datas = file.readlines()[2:]
            category_list = [line.strip().split()[0] for line in datas]
        return category_list

    # 格式{count:xx, data:["Anorak","Blazer","Blouse",...]}
    def build_category_label_json_file(self):
        json_name = "category_label.json"
        category_list = self.get_category_list()
        json_obj = {"count": int(len(category_list)), "data": category_list}
        ju.write_json_file(json_obj, self._json_dir, json_name)

    # 格式{count:289222,data:[{categoryNum:1,bbox:[x0,y0,width,height],path:"xxx/yyy/zz.jpg"},...]}
    def build_image_label_json_file(self):
        all_json_name = "image_label.json"
        bbox_file_path = self._base_dir + self._list_bbox_path
        image_path = self._base_dir + self._list_category_image_path
        with open(bbox_file_path, "r") as bb_image_file, open(image_path, "r") as image_file:
            category_dict = self.get_category_dict()
            bb_image_list = [line.split() for line in bb_image_file.readlines()]
            image_List = [line.split() for line in image_file.readlines()]
            all_list = []
            for n in range(2, len(bb_image_list)):
                path = image_List[n][0]
                bbox = [int(x) for x in bb_image_list[n][1:]]
                cateName = path.split("/")[1].split("_")[-1]
                cateNum = category_dict[cateName]
                jsonObj = {"path": path, "bbox": bbox, "categoryNum": cateNum}
                all_list.append(jsonObj)

            all_json = {"count": len(all_list), "data": all_list}
            ju.write_json_file(all_json, self._json_dir, all_json_name)

def set_parser():
    parser = argparse.ArgumentParser(description="this script build json data from deepfashion")
    parser.add_argument("-output", action="store", default="", help="output path for json file")
    parser.add_argument("-source", action="store", default="", help="base dir of deep fashion")
    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

# 两个参数一个-output, 表示输出目录, 一个-source 表示deepfashion目录
# 脚本用于生成json文件,包括属性列表json,所有图片数据json
def main():
    FLAGS = set_parser()
    dc = DeepFashionConverter()
    json_dir = FLAGS.output
    base_dir = FLAGS.source
    if json_dir != "":
        dc.set__json_dir(json_dir)
    if base_dir != "":
        dc.set__base_dir(base_dir)
    dc.build_category_label_json_file()
    dc.build_image_label_json_file()

if __name__ == '__main__':
    main()
