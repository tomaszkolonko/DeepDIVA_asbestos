#!/bin/bash

# ResNet18
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/fc_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 5

# ResNet18 --pretrained
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_pre \
    --output-folder /home/thomas.kolonko/fc_output_asbestos_resnet18_multi_pre --ignoregit \
    --lr 0.039918 --momentum 0.170826 --weight-decay 0.001980 --decay-lr 20 \
    --multi-run 5 --pretrained

# Densenet169
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
    --model-name densenet169 --epochs 50 --experiment-name tz_asbestos_densenet169 \
    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet169_multi --ignoregit \
    --lr 0.005812 --momentum 0.777249 --weight-decay 0.006999 --decay-lr 20 \
    --multi-run 5

# Densenet169 --pretrained
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL_CLEARED/ \
    --model-name densenet169 --epochs 50 --experiment-name tz_asbestos_densenet169_pre \
    --output-folder /home/thomas.kolonko/fc_output_asbestos_densenet169_multi --ignoregit \
    --lr 0.006347 --momentum 0.447591 --weight-decay 0.005180 --decay-lr 20 \
    --multi-run 5 --pretrained