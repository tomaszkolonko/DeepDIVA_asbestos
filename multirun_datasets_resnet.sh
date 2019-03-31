#!/bin/bash

## FINAL DATASET
##################

#
## ResNet18
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/f_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/f_output_asbestos_resnet18_multi_pre --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Densenet121
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
#    --output-folder /home/thomas.kolonko/f_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
#    --multi-run 3
#
## Densenet121 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
#    --output-folder /home/thomas.kolonko/f_output_asbestos_densenet121_multi_pre --ignoregit \
#    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Inception v3
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception \
#    --output-folder /home/thomas.kolonko/f_output_asbestos_inception_multi --ignoregit \
#    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
#    --multi-run 3
#
## Inception v3 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_pre \
#    --output-folder /home/thomas.kolonko/f_output_asbestos_inception_multi_pre --ignoregit \
#    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
#
### FINAL CLEARED DATASET
#########################
#
## ResNet18
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_resnet18_multi_pre --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Densenet121
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
#    --multi-run 3
#
## Densenet121 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet121_multi_pre --ignoregit \
#    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Inception v3
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_inception_multi --ignoregit \
#    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
#    --multi-run 3
#
## Inception v3 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_pre \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_inception_multi_pre --ignoregit \
#    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
#
### FINAL CLEARED HEAVY DATASET
###############################
#
## ResNet18
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/fch_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/fch_output_asbestos_resnet18_multi_pre --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Densenet121
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
#    --output-folder /home/thomas.kolonko/fch_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
#    --multi-run 3
#
## Densenet121 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
#    --output-folder /home/thomas.kolonko/fch_output_asbestos_densenet121_multi_pre --ignoregit \
#    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Inception v3
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception \
#    --output-folder /home/thomas.kolonko/fch_output_asbestos_inception_multi --ignoregit \
#    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
#    --multi-run 3
#
## Inception v3 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_pre \
#    --output-folder /home/thomas.kolonko/fch_output_asbestos_inception_multi_pre --ignoregit \
#    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
#
### FINAL CLEARED BALANCED DATASET
##################################
#
#
## ResNet18
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C_B/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/fcb_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C_B/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/fcb_output_asbestos_resnet18_multi_pre --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Densenet121
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C_B/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
#    --output-folder /home/thomas.kolonko/fcb_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
#    --multi-run 3
#
## Densenet121 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C_B/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
#    --output-folder /home/thomas.kolonko/fcb_output_asbestos_densenet121_multi_pre --ignoregit \
#    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Inception v3
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C_B/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception \
#    --output-folder /home/thomas.kolonko/fcb_output_asbestos_inception_multi --ignoregit \
#    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
#    --multi-run 3
#
## Inception v3 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_C_B/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_pre \
#    --output-folder /home/thomas.kolonko/fcb_output_asbestos_inception_multi_pre --ignoregit \
#    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
### FINAL CLEARED HEAVY BALANCED DATASET
########################################
#
## ResNet18
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/fchb_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/fchb_output_asbestos_resnet18_multi_pre --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Densenet121
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
#    --output-folder /home/thomas.kolonko/fchb_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
#    --multi-run 3
#
## Densenet121 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
#    --output-folder /home/thomas.kolonko/fchb_output_asbestos_densenet121_multi_pre --ignoregit \
#    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
## Inception v3
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception \
#    --output-folder /home/thomas.kolonko/fchb_output_asbestos_inception_multi --ignoregit \
#    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
#    --multi-run 3
#
## Inception v3 --pretrained
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_pre \
#    --output-folder /home/thomas.kolonko/fchb_output_asbestos_inception_multi_pre --ignoregit \
#    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
#    --multi-run 3 --pretrained


## FINAL EXTENDED DATASET
###########################

# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/fex_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18 --pretrained
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
    --output-folder /home/thomas.kolonko/fex_output_asbestos_resnet18_multi_pre --ignoregit \
    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
    --multi-run 3 --pretrained

# Densenet121
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
    --output-folder /home/thomas.kolonko/fex_output_asbestos_densenet121_multi --ignoregit \
    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
    --multi-run 3

# Densenet121 --pretrained
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
    --output-folder /home/thomas.kolonko/fex_output_asbestos_densenet121_multi_pre --ignoregit \
    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
    --multi-run 3 --pretrained

# Inception v3
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception \
    --output-folder /home/thomas.kolonko/fex_output_asbestos_inception_multi --ignoregit \
    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
    --multi-run 3

# Inception v3 --pretrained
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_pre \
    --output-folder /home/thomas.kolonko/fex_output_asbestos_inception_multi_pre --ignoregit \
    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
    --multi-run 3 --pretrained