# Train
# Train
# source /usr/local/miniconda/etc/profile.d/conda.sh
# conda activate fewx
# cd /home/user/ws/FewX/SUBT/AirDet
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:42431 \
# 	--config-file configs/fsod/R_50_C4_1x.yaml 2>&1 | tee log/fsod_train_log.txt

# Finetune
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco10.yaml MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_train_log.txt
### ----------------- SUBT Testing Part ------------------------- ###
# Test
# SUBT 1
 
# if[[ $strA =~ $strB ]]
# rm support_dir/support_feature.pkl
CUDA_VISIBLE_DEVICES=1 python3 test_final.py --input ./examples/final-circuit/UGV3/23_left/ --output ./results/final-circuit-101/UGV3/23_left/ --confidence-threshold 0.9 --config-file configs/fsod/test_R_101_subt3_final.yaml
CUDA_VISIBLE_DEVICES=1 python3 test_final.py --input ./examples/final-circuit/UGV3/28_back/ --output ./results/final-circuit-101/UGV3/28_back/ --confidence-threshold 0.9 --config-file configs/fsod/test_R_101_subt3_final.yaml
CUDA_VISIBLE_DEVICES=1 python3 test_final.py --input ./examples/final-circuit/UGV3/28_right/ --output ./results/final-circuit-101/UGV3/28_right/ --confidence-threshold 0.9 --config-file configs/fsod/test_R_101_subt3_final.yaml

# # COCO5
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_101_C4_1x_coco5.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_101_C4_1x/model_final.pth 2>&1 | tee log/fsod_test_log_coco5.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_101_C4_1x_coco5.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_101_C4_1x/model_final.pth 2>&1 | tee log/fsod_test_log_coco5.txt

# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_101_C4_1x_subt3.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_101_C4_1x/model_final.pth DATASETS.TEST "('val_e_6',)" TEST.VIS_DIR 'vis/val_e_6_3shot' TEST.VIS_THRESH 0.8 2>&1 | tee log/fsod_test_log_subt3_$i.txt

# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_101_C4_1x_subt3.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_101_C4_1x/model_final.pth DATASETS.TEST "('val_e_6',)" TEST.VIS_DIR 'vis/val_e_6_3shot' TEST.VIS_THRESH 0.8 2>&1 | tee log/fsod_test_log_subt3_$i.txt
