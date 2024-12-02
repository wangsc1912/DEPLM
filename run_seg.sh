python train_segmentation.py --learning_rate 0.02 \
    --lr_decay 0.5 \
    --step_size 28 \
    --epoch 101 \
    --batch_size 256 \
    --gpu 1 \
    --r0 0.1 \
    --r1 0.3 \
    --sparsity 0.5 \
    --c_prune_rate 3.2 \
    --feat1 375 \
    --num_feat 750 \
    --quant_bit 5 \
    --model model_part_seg \
    --folder segmentation \
    --hard_mode batch --noise 0.02

