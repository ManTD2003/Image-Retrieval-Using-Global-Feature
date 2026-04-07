python evaluate.py \
    --model vit_s16 \
    --train ./cifar100_features/vit_s16_cifar100_train.pkl \
    --test  ./cifar100_features/vit_s16_cifar100_test.pkl \
    --K 1 5 10 \
    --savedir ./results/
