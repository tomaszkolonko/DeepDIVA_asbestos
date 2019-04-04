#!/bin/bash

# ResNet18
python ./template/RunMe.py --runner-class image_classification_full_image \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet18_448 --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/f_448_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification_full_image \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet18_896 --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/f_896_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification_full_image \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet18_1024 --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/f_1024_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 20 \
    --multi-run 3

# ResNet18
python ./template/RunMe.py --runner-class image_classification_random_nine \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet_sixteen --epochs 150 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/f_partfilt_randomnine_output_asbestos_resnet18_multi --ignoregit \
    --lr 0.033678 --momentum 0.952630 --weight-decay 0.007518 --decay-lr 40 \
    --multi-run 3