import pandas as pd
from os.path import join, isdir
from os import mkdir, makedirs
import cv2

def generate_new_df(df_path, cls, path):
    if not isdir(path): makedirs(path)
    support_df = pd.read_pickle(df_path)
    new_df = {}
    
    new_df['support_box'] = []
    new_df['category_id'] = []
    new_df['image_id'] = []
    new_df['id'] = []
    new_df['file_path'] = []
    cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
    for index, support_img_df in cls_df.iterrows():
        img_path = support_img_df['file_path']
        im = cv2.imread(join('./SUBT',img_path[2:]))
        new_path = join(path, '{:04d}.jpg'.format(index))
        cv2.imwrite(new_path, im)
        new_df['support_box'].append(support_img_df['support_box'])
        new_df['category_id'].append(support_img_df['category_id'])
        new_df['image_id'].append(support_img_df['image_id'])
        new_df['id'].append(support_img_df['id'])
        new_df['file_path'].append(join('./',new_path[7:]))
    new_support = pd.DataFrame.from_dict(new_df)
    return new_support

if __name__ == '__main__':
    df_path = './SUBT/use/{}/3_shot_support_df.pkl'.format('val_e_4')
    new_support = generate_new_df(df_path,2,"./SUBT/final_support/3_shot_rope_e4")
    new_support.to_pickle(join("./SUBT/final_support", "3_shot_rope_e4_df.pkl"))