import os

base_path='./use'
dicti={}
for test in os.listdir(base_path):
    dicti[test] = (test+"/JPEGImages", test+"/new_annotations/{}.json".format(test))

print(dicti)