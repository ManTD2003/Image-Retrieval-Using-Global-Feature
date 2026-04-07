python evaluate.py \
    --model vit_s16 \
    --train ./cifar10_features/vit_s16_cifar10_train.pkl \
    --test ./cifar10_features/vit_s16_cifar10_test.pkl \
    --K 1 5 10 \
    --savedir ./results/ 
