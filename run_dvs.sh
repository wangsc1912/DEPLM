python train_classification_dvs.py \
--epoch 200 \
--batch_size 512 \
--learning_rate 0.01 \
--gpu 0 \
--folder dvs_june_sup \
--log_dir sparse_0.5 \
--c_prune_rate 2.38 \
--num_feat 1219 \
--dataset dvsgestures \
--num_category 10 \
--model model_cls_rand \
--sa_iter 1,1,1 \
--sparsity 0.5 \
--scale 1. \
--quantize full \
--radius 0.18 \
--radius_mul 0.45 \
--denoise_event \
--hard_mode batch