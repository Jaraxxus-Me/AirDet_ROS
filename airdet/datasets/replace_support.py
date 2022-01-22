import pandas as pd
import os
import json
from pycocotools.coco import COCO
import random

def getImgId(coco, img_name):
    res_id = []
    for im_id in list(coco.imgs.keys()):
        if coco.loadImgs(im_id)[0]['file_name'].split('/')[-1].split('.')[0] in img_name:
            res_id.append(im_id)
    return res_id


def new_json(coco, basepath, shot, cls_id, target_img=None):
    assert shot == len(target_img)
    origin_json = os.path.join(basepath, 'final_split_subt_{}_shot_instances_train.json'.format(shot))
    with open(origin_json, 'r') as f:
        origin_content = json.load(f)
        save_images = origin_content['images']
        save_categories = origin_content['categories']
        origin_anno = origin_content['annotations']
    
    new_anno = []
    for anno in origin_anno:
        if not anno['category_id'] == cls_id:
            new_anno.append(anno)

    img_cls_dict = {}
    for new_ann in new_anno:
        id = new_ann['id']
        category_id = new_ann['category_id']
        if category_id in img_cls_dict.keys():
            img_cls_dict[category_id] += 1
        else:
            img_cls_dict[category_id] = 1

    all_cls_dict = {}
    for category_id, num in img_cls_dict.items():
        if category_id in all_cls_dict.keys():
            all_cls_dict[category_id] += num
        else:
            all_cls_dict[category_id] = num
    new = []
    cls_ims = getImgId(coco, target_img)
    for id in cls_ims:
        img_cls_dict = {}
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        skip_flag = False
        if len(anns) != 1:
            continue
        for ann in anns:
            area = ann['area']
            category_id = ann['category_id']
            id = ann['id']
            
            if category_id in img_cls_dict.keys():
                img_cls_dict[category_id] += 1
            else:
                img_cls_dict[category_id] = 1
            
            # filter images with small boxes
            if area < 64 * 64 or area > 224 * 224:
                skip_flag = True
                
            if category_id in all_cls_dict.keys():
                if all_cls_dict[category_id] == shot:
                    skip_flag = True

        if skip_flag:
            continue
        else:
            for ann in anns:
                new_anno.append(ann)
                new.append(ann)
            for category_id, num in img_cls_dict.items():
                if category_id in all_cls_dict.keys():
                    all_cls_dict[category_id] += num
                else:
                    all_cls_dict[category_id] = num    
    print(len(new_anno))
    print(new)
    print(sorted(all_cls_dict.items(), key = lambda kv:(kv[1], kv[0]))) 
    
    dataset_split = {
        'images': save_images,
        'annotations': new_anno,
        'categories': save_categories}
    return dataset_split

if __name__ == '__main__':
    test_seq = 'val_c_1'
    annFile = './SUBT/use/{}/{}.json'.format(test_seq, test_seq)
    base_path = './SUBT/use/{}/new_annotations'.format(test_seq)
    coco = COCO(annFile)
    new_dict = new_json(coco, base_path, 3, 8, ['c_1_03190', 'c_1_05260', 'c_1_06990'])
    split_file = './SUBT/use/{}/new_annotations/new_split_subt_{}_shot_instances_train.json'.format(test_seq,3)
    with open(split_file, 'w') as f:
        json.dump(new_dict, f)
