category_map = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush'
}


import torch
import json
import os
import cv2
from detectron2.utils.visualizer import ColorMode,Visualizer
from detectron2.structures.instances import Instances

def Visualize(input, output, dir, train_metadata, thresh):
    im = cv2.imread(input['file_name'])
    name = input['file_name'].split("/")[-1]
    meta = train_metadata
    # meta.thing_colors = []
    for i in range(len(train_metadata.thing_classes)):
        meta.thing_colors[i]= [0,255,0]
    v = Visualizer(im[:, :, ::-1],
                metadata=meta,
                scale=0.5,
                instance_mode=ColorMode.SEGMENTATION
                 )
    all_instances = output["instances"].to("cpu")
    if thresh >= 1:
        # Top K mode
        draw_instances =  Instances(all_instances.image_size)
        draw_instances.pred_boxes = all_instances.pred_boxes[0:thresh]
        draw_instances.scores = all_instances.scores[0:thresh]
        draw_instances.pred_classes = all_instances.pred_classes[0:thresh]
    if thresh < 1:
        # Score Threshhold mode
        draw_instances =  Instances(all_instances.image_size)
        n=0
        for score in all_instances.scores:
            if score >= thresh:
                n+=1
        if n>1:
            n=2
        draw_instances.pred_boxes = all_instances.pred_boxes[0:n]
        draw_instances.scores = all_instances.scores[0:n]
        draw_instances.pred_classes = all_instances.pred_classes[0:n]
    out = v.draw_instance_predictions(draw_instances)
    # boxes = v._convert_boxes(output["instances"].pred_boxes.to('cpu'))
    # for box in boxes:
    #     box = (round(box[0]), round(box[1]), round(box[2]) - round(box[0]), round(box[3] - box[1]))
    #     out = v.draw_text(f"{box[2:4]}", (box[0], box[1]))
    cv2.imwrite(os.path.join(dir,name), out.get_image()[:, :, ::-1])



# all_res_path="./output/finetune_dir/R_50_C4_1x/inference/instances_predictions.pth"
# annos="./datasets/coco/annotations/instances_val2017.json"
# base_img_path = './datasets/coco/val2017'
# ann_f = open(annos, 'r')
# ann_json = json.load(ann_f)
# find = {}
# for anno in ann_json['images']:
#     find[anno['id']] = anno['file_name']
#     find[anno['file_name']] = (anno['height'], anno['width'])

# all_res = torch.load(all_res_path)
# for res in all_res: 
#     idx = res['image_id']
#     img_file = find[idx]
#     img_size = find[img_file]
#     im = cv2.imread(os.path.join(base_img_path,img_file))
#     outputs = res['instances']
#     scores=[]
#     pred_boxes=[]
#     pre_cls=[]
#     for output in outputs:
#         if output["score"]>0.7:
#             scores.append(output["score"])
#             pred_boxes.append(output["bbox"])
#             pre_cls.append(category_map[output["category_id"]])
#     if len(scores):
#         print(str(idx)+':'+img_file)  
#         i = Instances(img_size)
#         i.set('scores',scores)
#         i.set('pred_classes',pre_cls)
#         i.set('pred_boxes',pred_boxes)
#         v = Visualizer(im[:, :, ::-1],
#                     #    metadata=meta_img, 
#                     #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#         )
#         out = v.draw_instance_predictions(i)
#         cv2.imwrite('vis/'+img_file, out.get_image()[:, :, ::-1])