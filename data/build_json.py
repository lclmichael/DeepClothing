# encoding="utf8"
# Author="LclMichael"

import os
import json
import random
import argparse

DEFAULT_OUTPUT_DIR = "./json/"

# deep fashion默认目录
category_dir = "E:/DataSet/DeepFashion/Category and Attribute Prediction Benchmark/Anno/"


def write_json_file(jsonData, output_dir, file_name):
    print("start building " + file_name)
    try:
        with open(os.path.join(output_dir, file_name), "w+") as file:
            json.dump(jsonData, file)
    except Exception as e:
        print("build " + file_name + " fail.")
        print(e)
    else:
        print("build " + file_name + " success, filePath : " + output_dir + file_name)

# 从deepfashion的list_category_cloth.txt中获取类别字典
def bulid_category_dict():
    categoryFilePath = category_dir + "list_category_cloth.txt"
    with open(categoryFilePath, "r") as cFile:
        datas = cFile.readlines()[2:]
        categoryDict = {line.strip().split()[0]: index for index, line in enumerate(datas)}
        cFile.close()
    return categoryDict

# 从deepfashion的list_category_cloth.txt中获取类别数组
def bulid_category_list():
    categoryFilePath = category_dir + "list_category_cloth.txt"
    with open(categoryFilePath, "r") as cFile:
        datas = cFile.readlines()[2:]
        categoryList = [line.strip().split()[0] for line in datas]
    return categoryList

# 格式{count:289222,data:[{categoryNum:1,bbox:[x0,y0,width,height],path:"xxx/yyy/zz.jpg"},...]}
def build_image_label_to_json_file(output_dir=DEFAULT_OUTPUT_DIR):
    allfile_name = "image_label_all.json"
    trainfile_name = "image_label_train.json"
    validatefile_name = "image_label_validate.json"
    testfile_name = "image_label_test.json"
    categoryDict = bulid_category_dict()
    bboxImageFilePath = category_dir + "list_bbox.txt"
    categoryImageFilePath = category_dir + "list_category_img.txt"
    bbImageFile = open(bboxImageFilePath, "r")
    cImageFile = open(categoryImageFilePath, "r")

    bbImageList = [line.split() for line in bbImageFile.readlines()]
    cImageList = [line.split() for line in cImageFile.readlines()]
    allList = []
    for n in range(2, len(bbImageList)):
        path = cImageList[n][0]
        bbox = [int(x) for x in bbImageList[n][1:]]
        cateName = path.split("/")[1].split("_")[-1]
        cateNum = categoryDict[cateName]
        jsonObj = {"path": path, "bbox": bbox, "categoryNum": cateNum}
        allList.append(jsonObj)

    allJson = {"count": len(allList), "data": allList}
    write_json_file(allJson, output_dir, allfile_name)

    # build data for train(90%), validate(5%), test(5%)
    allCount = allJson["count"]
    validateList = [allList.pop(allList.index(random.choice(allList))) for _ in range(int(allCount * 0.05))]
    testList = [allList.pop(allList.index(random.choice(allList))) for _ in range(int(allCount * 0.05))]

    write_json_file({"count": len(allList), "data": allList}, output_dir, trainfile_name)
    write_json_file({"count": len(validateList), "data": validateList}, output_dir, validatefile_name)
    write_json_file({"count": len(testList), "data": testList}, output_dir, testfile_name)

    bbImageFile.close()
    cImageFile.close()

# 格式{count:xx, data:["Anorak","Blazer","Blouse",...]}
def build_category_label_to_json_file(output_dir=DEFAULT_OUTPUT_DIR):
    file_name = "category_label.json"
    cateList = bulid_category_list()
    write_json_file({"count": int(len(cateList)), "data": cateList}, output_dir, file_name)

def set_parser():
    parser = argparse.ArgumentParser(description="this script build json data from deepfashion")
    parser.add_argument("-o", action="store", default="./json/", help="output path for file")
    parser.add_argument("-s", action="store", default="", help="base dir of deep fashion category")
    FLAGS, unknown = parser.parse_known_args()
    return FLAGS

# 两个参数一个-o, 表示输出目录, 一个-s 表示deepfashion目录
# 脚本用于生成json文件,包括属性列表json,所有图片数据json,
# 以及基于此90%, 5%, 5%三个比例分别生成的训练以及验证,泛化的文件
def main():
    FLAGS = set_parser()
    output_dir = FLAGS.o
    global category_dir
    if FLAGS.s != "":
        category_dir = FLAGS.s

    build_category_label_to_json_file(output_dir)
    build_image_label_to_json_file(output_dir)

if __name__ == '__main__':
    main()
