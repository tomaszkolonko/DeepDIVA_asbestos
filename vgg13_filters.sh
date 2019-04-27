#!/bin/bash

## A
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_a --epochs 50 --experiment-name vgg13_bn_a_16 \
#    --output-folder /home/thomas.kolonko/f_vgg13_a_16 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_a --epochs 50 --experiment-name vgg13_bn_a_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_a --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
## B
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_b --epochs 50 --experiment-name vgg13_bn_b_16 \
#    --output-folder /home/thomas.kolonko/f_vgg13_b_16 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_b --epochs 50 --experiment-name vgg13_bn_b_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_b --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
## C
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_c --epochs 50 --experiment-name vgg13_bn_c \
#    --output-folder /home/thomas.kolonko/f_vgg13_c --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_c --epochs 50 --experiment-name vgg13_bn_c_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_c --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
# D
###

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/f_vgg13_d_16_optimized_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_d --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
## E
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_e --epochs 50 --experiment-name vgg13_bn_e \
#    --output-folder /home/thomas.kolonko/f_vgg13_e --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_e --epochs 50 --experiment-name vgg13_bn_e_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_e --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
## F
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_f --epochs 50 --experiment-name vgg13_bn_f \
#    --output-folder /home/thomas.kolonko/f_vgg13_f --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_f --epochs 50 --experiment-name vgg13_bn_f_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_f --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained


# G
###

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_g_16_optimized_crop \
    --output-folder /home/thomas.kolonko/f_vgg13_g_16_optimized_crop --ignoregit --lr 0.1 \
    --momentum 0.499036 --weight-decay 0.00001 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn_g --epochs 170 --experiment-name vgg13_bn_g_16_optimized_crop_long \
    --output-folder /home/thomas.kolonko/f_vgg13_g_16_optimized_crop_long --ignoregit --lr 0.1 \
    --momentum 0.499036 --weight-decay 0.00001 --decay-lr 50 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_g --epochs 170 --experiment-name vgg13_bn_g_16_optimized_crop_long \
    --output-folder /home/thomas.kolonko/fex_vgg13_g_16_optimized_crop_long --ignoregit --lr 0.1 \
    --momentum 0.499036 --weight-decay 0.00001 --decay-lr 50 --multi-run 3

#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_g_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_g --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
## H
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_h --epochs 50 --experiment-name vgg13_bn_h \
#    --output-folder /home/thomas.kolonko/f_vgg13_h --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_h --epochs 50 --experiment-name vgg13_bn_h_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_h --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
## I
####
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_i --epochs 50 --experiment-name vgg13_bn_i \
#    --output-folder /home/thomas.kolonko/f_vgg13_i --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_i --epochs 50 --experiment-name vgg13_bn_i_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_i --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained

# J
###

#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_j --epochs 50 --experiment-name vgg13_bn_j \
#    --output-folder /home/thomas.kolonko/f_vgg13_j --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_bn_i --epochs 50 --experiment-name vgg13_bn_i_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_i --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained