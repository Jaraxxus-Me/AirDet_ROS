import pandas as pd
from os.path import join, isdir
from os import mkdir, makedirs
import cv2

def generate_new_df(df_paths, path):
    if not isdir(path): makedirs(path)
    new_df = {}
    new_df['support_box'] = []
    new_df['category_id'] = []
    new_df['image_id'] = []
    new_df['id'] = []
    new_df['file_path'] = []
    index = 0
    for df_path in df_paths:
        support_df = pd.read_pickle(df_path)
        for i, support_img_df in support_df.iterrows():
            img_path = support_img_df['file_path']
            im = cv2.imread(join('./SUBT',img_path[2:]))
            new_path = join(path, '{:04d}.jpg'.format(index))
            cv2.imwrite(new_path, im)
            new_df['support_box'].append(support_img_df['support_box'])
            new_df['category_id'].append(support_img_df['category_id'])
            new_df['image_id'].append(support_img_df['image_id'])
            new_df['id'].append(support_img_df['id'])
            new_df['file_path'].append(join('./',new_path[7:]))
            index+=1
    new_support = pd.DataFrame.from_dict(new_df)
    return new_support

if __name__ == '__main__':
    df_paths = []
    cls = ['rope', 'drill', 'backpack']
    df_paths.append('./SUBT/final_support/3_shot_{}_df.pkl'.format('rope_e4'))
    df_paths.append('./SUBT/final_support/3_shot_{}_df.pkl'.format('drill_g23'))
    df_paths.append('./SUBT/final_support/3_shot_{}_df.pkl'.format('backpack_g39'))
    # df_paths.append('./SUBT/final_support/3_shot_{}_df.pkl'.format('val_g_25'))
    new_support = generate_new_df(df_paths,"./SUBT/final_support/3_shot_{}_{}_{}".format(cls[0],cls[1],cls[2]))
    new_support.to_pickle(join("./SUBT/final_support", "3_shot_{}_{}_{}_df.pkl".format(cls[0],cls[1],cls[2])))