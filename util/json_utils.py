#encoding=utf8
#Author=LclMichael

import os
import json

# read json file
def read_json_file(json_path, encoding = "utf8"):
    with open(json_path, encoding=encoding) as file:
        return json.load(file)

# write json file
def write_json_file(jsonData, output_dir, file_name):
    print("start building " + file_name)
    full_path = os.path.join(output_dir, file_name)
    try:
        with open(full_path, "w+") as file:
            json.dump(jsonData, file)
    except Exception as e:
        print("build " + file_name + " fail.")
        print(e)
    else:
        print("build " + file_name + " success")
    finally:
        print("file path : " + full_path)
