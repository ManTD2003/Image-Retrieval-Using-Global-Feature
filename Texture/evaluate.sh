python evaluate.py \
    --model vit_s16 \
    --train ./texture_features/vit_s16_texture_train.pkl \
    --valid ./texture_features/vit_s16_texture_valid.pkl \
    --K 1 5 10 \
    --savedir ./results/
