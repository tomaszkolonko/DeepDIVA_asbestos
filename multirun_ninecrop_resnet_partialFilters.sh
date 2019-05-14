#!/bin/bash

# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name resnet_sixteen --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/new_fch_partfilt_16_simple_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name resnet_sixteen --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/new_fchb_partfilt_16_simple_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name resnet_sixteen --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/new_fex_partfilt_16_simple_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3


# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name resnet_two --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/new_fch_partfilt_2_simple_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name resnet_two --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/new_fchb_partfilt_2_simple_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification\
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name resnet_two --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/new_fex_partfilt_2_simple_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3








python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fch_vgg13_d_16_simple_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fchb_vgg13_d_16_simple_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fex_vgg13_d_16_simple_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3



python ./template/RunMe.py --runner-class image_classification_five_crop \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fch_vgg13_d_16_fivecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_five_crop \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fchb_vgg13_d_16_fivecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_five_crop \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fex_vgg13_d_16_fivecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3



python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fch_vgg13_d_16_ninecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fchb_vgg13_d_16_ninecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_d --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fex_vgg13_d_16_ninecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3









python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fch_vgg13_g_16_simple_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fchb_vgg13_g_16_simple_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fex_vgg13_g_16_simple_crop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3



python ./template/RunMe.py --runner-class image_classification_five_crop \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fch_vgg13_g_16_fivecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_five_crop \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fchb_vgg13_g_16_fivecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_five_crop \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fex_vgg13_g_16_fivecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3



python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fch_vgg13_g_16_ninecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL_CH_B/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fchb_vgg13_g_16_ninecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL_EXTENDED/ \
    --model-name vgg13_bn_g --epochs 50 --experiment-name vgg13_bn_d_16_optimized_crop \
    --output-folder /home/thomas.kolonko/new_fex_vgg13_g_16_ninecrop --ignoregit --lr 0.077457 \
    --momentum 0.310723 --weight-decay 0.009106 --decay-lr 20 --multi-run 3

## ResNet18
#python ./template/RunMe.py --runner-class image_classification_random_nine \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/fc_fivecrop_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification_random_nine \
#    --dataset-folder /home/thomas.kolonko/FINAL_C/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/fc_fivecrop_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained
#
#
## ResNet18
#python ./template/RunMe.py --runner-class image_classification_random_nine \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
#    --output-folder /home/thomas.kolonko/fch_fivecrop_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
#    --multi-run 3
#
## ResNet18 --pretrained
#python ./template/RunMe.py --runner-class image_classification_random_nine \
#    --dataset-folder /home/thomas.kolonko/FINAL_CH/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
#    --output-folder /home/thomas.kolonko/fch_fivecrop_output_asbestos_resnet18_multi --ignoregit \
#    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
#    --multi-run 3 --pretrained

## Densenet121
#python ./template/RunMe.py --runner-class image_classification_five_crop \
#    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121 \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.035925 --momentum 0.057618 --weight-decay 0.009241 --decay-lr 20 \
#    --multi-run 5
#
## Densenet121 --pretrained
#python ./template/RunMe.py --runner-class image_classification_five_crop \
#    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_pre \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet121_multi --ignoregit \
#    --lr 0.018489 --momentum 0.369998 --weight-decay 0.004963 --decay-lr 20 \
#    --multi-run 5 --pretrained
#
## Inception v3
#python ./template/RunMe.py --runner-class image_classification_five_crop \
#    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
#    --model-name densenet169 --epochs 50 --experiment-name tz_asbestos_densenet169 \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet169_multi --ignoregit \
#    --lr 0.070046 --momentum 0.910505 --weight-decay 0.006943 --decay-lr 20 \
#    --multi-run 5
#
## Inception v3 --pretrained
#python ./template/RunMe.py --runner-class image_classification_five_crop \
#    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
#    --model-name densenet169 --epochs 50 --experiment-name tz_asbestos_densenet169_pre \
#    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet169_multi --ignoregit \
#    --lr 0.029269 --momentum 0.0 --weight-decay 0.006320 --decay-lr 20 \
#    --multi-run 5 --pretrained