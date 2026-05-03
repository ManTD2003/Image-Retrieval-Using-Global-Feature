# Image Retrieval Using Global Features

## About

Content-Based Image Retrieval (CBIR) is a computer vision task that involves querying an image database to find the top **K** most similar images to a given input (query) image, based on a specific similarity metric. 

The goal of this project is to explore the possibilities, benefits, and potential drawbacks of applying Vision Transformers to solve image retrieval problems using global feature representations.

## Quick Start

### 1) Create and activate environment


```bash
conda create -n image-retrieval python=3.10 -y
conda activate image-retrieval
pip install -r requirements.txt
```

### 2) Run in order

Run each dataset strictly in this order: `train -> extract_features -> evaluate`.
Do not run the next command until the previous command has finished and produced its output file/model.

For each dataset, execute step-by-step:
1. Run `train_metric.py` and wait until training is done (model file is created in `model_save/`).
2. Run `extract_features.py` only after the model file exists.
3. Run `evaluate.py` only after feature file(s) are created.

#### CIFAR10

```bash
cd CIFAR10
python train_metric.py --models vit_s16 --batch_size 128

python extract_features.py --model vit_s16 --weight ./model_save/vit_s16_cifar10.pth

python evaluate.py --model vit_s16 --train ./cifar10_features/vit_s16_cifar10_train.pkl --test ./cifar10_features/vit_s16_cifar10_test.pkl --K 1 5 10 --savedir ./results/
```

#### CIFAR100

```bash
cd CIFAR100
python train_metric.py --models vit_s16 --batch_size 128

python extract_features.py --model vit_s16 --weight ./model_save/vit_s16_cifar100.pth

python evaluate.py --model vit_s16 --train ./cifar100_features/vit_s16_cifar100_train.pkl --test ./cifar100_features/vit_s16_cifar100_test.pkl --K 1 5 10 --savedir ./results/
```

#### Texture

```bash
cd Texture
python train_metric.py --models vit_s16 --batch_size 128

python extract_features.py --model vit_s16 --weight ./model_save/vit_s16_texture.pth

python evaluate.py --model vit_s16 --features ./texture_features/vit_s16_texture.pkl --K 1 5 10 --savedir ./results/
```
