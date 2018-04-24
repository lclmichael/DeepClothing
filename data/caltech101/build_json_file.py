# encoding=utf8
# Author=LclMichael

import time

from deepclothing.data.caltech101.json_data import JsonDataTools

def main():

    start = time.time()
    jdt = JsonDataTools()
    jdt.build_json_file()
    print("cost time {}".format(time.time() - start))
    pass
    
if __name__ == "__main__":
    main()