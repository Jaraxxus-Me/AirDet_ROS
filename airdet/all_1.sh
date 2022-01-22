# Train
# source /usr/local/miniconda/etc/profile.d/conda.sh
# conda activate fewx
# cd /home/user/ws/FewX/v7_4_conv_res2/
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/R_50_C4_1x.yaml 2>&1 | tee log/fsod_train_log.txt

CUDA_VISIBLE_DEVICES=1 python3 test_final.py --input ./examples/final-circuit/UGV3/21_right/ --output ./results/final-circuit-101/UGV3/21_right/ --confidence-threshold 0.9 --config-file configs/fsod/test_R_101_subt3_final.yaml
CUDA_VISIBLE_DEVICES=1 python3 test_final.py --input ./examples/final-circuit/UGV3/22_front/ --output ./results/final-circuit-101/UGV3/22_front/ --confidence-threshold 0.9 --config-file configs/fsod/test_R_101_subt3_final.yaml
CUDA_VISIBLE_DEVICES=1 python3 test_final.py --input ./examples/final-circuit/UGV3/22_left/ --output ./results/final-circuit-101/UGV3/22_left/ --confidence-threshold 0.9 --config-file configs/fsod/test_R_101_subt3_final.yaml

# Finetune
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco10.yaml MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_train_log.txt

# FIND 1
# for check in 0111999 0112499 0112999 0113499 0113999 0114499 0114999 0115499 0115999 0116499 0116999 0117499 0117999 0118499 0118999 0119499 0119999
# do
# # COCO1: 115499, 114999, 113999, 1134999, 112999,112499
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco1.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_$check.pth 2>&1 | tee log/fsod_finetune_test_log_coco1_$check.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco1.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_$check.pth 2>&1 | tee log/fsod_finetune_test_log_coco1_$check.txt
# done

# # FIND 2
# for check in 0115499 0114999 0113999 0113499 0112999 0112499
# do
# # COCO2:
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco2.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_$check.pth 2>&1 | tee log/fsod_finetune_test_log_coco2_$check.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco2.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_$check.pth 2>&1 | tee log/fsod_finetune_test_log_coco2_$check.txt
# done

# # FIND 3
# for check in 0115499 0114999 0113999 0113499 0112999 0112499
# do
# # COCO3: 115499, 114999, 113999, 1134999, 112999,112499
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 --dist-url tcp://127.0.0.1:49141 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco3.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_$check.pth 2>&1 | tee log/fsod_finetune_test_log_coco3_$check.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:49141\
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco3.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_$check.pth 2>&1 | tee log/fsod_finetune_test_log_coco3_$check.txt
# done
# check = 112999
# Test
# COCO1
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco1.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco1.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco1.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco1.txt
# COCO2
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco2.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco2.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco2.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco2.txt
# # COCO3
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco3.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco3.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco3.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco3.txt
# # COCO4
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco4.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco4.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco4.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco4.txt
# # COCO5
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco5.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco5.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco5.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco5.txt
# COCO10
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco10.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco10.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_coco10.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/fsod_finetune_test_log_coco10.txt

### ----------------- VOC Testing Part ------------------------- ###

# VOC3
# rm support_dir/support_feature.pkl
# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_voc2.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/voc/fsod_finetune_test_log_voc2.txt

# CUDA_VISIBLE_DEVICES=0 python3 fsod_train_net.py --num-gpus 1 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_voc2.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/R_50_C4_1x/model_final.pth 2>&1 | tee log/voc/fsod_finetune_test_log_voc2.txt