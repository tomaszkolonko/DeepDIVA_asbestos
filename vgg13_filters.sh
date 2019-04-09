#!/bin/bash

# A
###

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_a --epochs 50 --experiment-name vgg13_bn_a \
    --output-folder /home/thomas.kolonko/f_vgg13_a --ignoregit --lr 0.054173 \
    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_a --epochs 50 --experiment-name vgg13_bn_a_pre \
    --output-folder /home/thomas.kolonko/f_vgg13_a --ignoregit --lr 0.093533 \
    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained


# B
###

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_b --epochs 50 --experiment-name vgg13_bn_b \
    --output-folder /home/thomas.kolonko/f_vgg13_b --ignoregit --lr 0.054173 \
    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_b --epochs 50 --experiment-name vgg13_bn_b_pre \
    --output-folder /home/thomas.kolonko/f_vgg13_b --ignoregit --lr 0.093533 \
    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained


# C
###

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_c --epochs 50 --experiment-name vgg13_bn_c \
    --output-folder /home/thomas.kolonko/f_vgg13_c --ignoregit --lr 0.054173 \
    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_c --epochs 50 --experiment-name vgg13_bn_c_pre \
    --output-folder /home/thomas.kolonko/f_vgg13_c --ignoregit --lr 0.093533 \
    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained