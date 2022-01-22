# COCO
DATA_ROOT=/ocean/projects/tra190016p/bli5/coco2017
cd coco
# Train
ln -s $DATA_ROOT/train2017 ./
ln -s $DATA_ROOT/val2017 ./
ln -s $DATA_ROOT/annotations ./

OTHER_ROOT=/jet/home/bli5/ws/FIND/datasets/coco
# Train
ln -s $OTHER_ROOT/new_annotations ./
ln -s $OTHER_ROOT/support ./
ln -s $OTHER_ROOT/1_shot_support ./
ln -s $OTHER_ROOT/2_shot_support ./
ln -s $OTHER_ROOT/3_shot_support ./
ln -s $OTHER_ROOT/4_shot_support ./
ln -s $OTHER_ROOT/5_shot_support ./
ln -s $OTHER_ROOT/10_shot_support ./
cp $OTHER_ROOT/train_support_df.pkl ./train_support_df.pkl
# Test
cp $OTHER_ROOT/1_shot_support_df.pkl ./1_shot_support_df.pkl
cp $OTHER_ROOT/2_shot_support_df.pkl ./2_shot_support_df.pkl
cp $OTHER_ROOT/3_shot_support_df.pkl ./3_shot_support_df.pkl
cp $OTHER_ROOT/4_shot_support_df.pkl ./4_shot_support_df.pkl
cp $OTHER_ROOT/5_shot_support_df.pkl ./5_shot_support_df.pkl
# cp $OTHER_ROOT/10_shot_support_df.pkl ./10_shot_support_df.pkl

# SUBT
SUBT_ROOT=/data/datasets/bowenli/CVPR2022/SUBT
cd SUBT

ln -s $SUBT_ROOT/JPEGImages ./
ln -s $SUBT_ROOT/annotations ./

mkdir use
python3 create_test.py
python3 1_split_filter.py ./ 
#!/bin/bash
cd use
for i in `ls`;do
ln -s $SUBT_ROOT/JPEGImages $i
# echo $i
done 
cd ..

# cd use
# for i in `ls`;do
# rm $i/new_annotations/final_split_subt_1_shot_instances_train.json
# rm $i/new_annotations/final_split_subt_2_shot_instances_train.json
# # echo $i
# done 
# cd ..

python3 3_gen_support_pool.py ./
python3 6_voc_few_shot.py ./
python3 4_gen_support_pool_10_shot.py ./
