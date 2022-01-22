import os
import json
# from tqdm import tqdm

anno_path = './annotations'
all_annos = os.listdir(anno_path)
for anno in all_annos:
    ann = open(os.path.join(anno_path, anno), 'r')
    content = json.load(ann)
    annos = content['annotations']
    cate = {}
    for a in annos:
        ca = a['category_id']
        if ca not in cate:
            cate[ca]=1
        else:
            cate[ca]+=1
    if len(annos)>=20:
        if not os.path.isdir(os.path.join('./',anno[:-5])):
            os.mkdir(os.path.join('./use',anno[:-5]))
        with open(os.path.join('./use',anno[:-5], anno), 'w') as f:
            json.dump(content, f)
        with open(os.path.join('./use',anno[:-5],'cate.json'), 'w') as f:
            json.dump(cate, f)

