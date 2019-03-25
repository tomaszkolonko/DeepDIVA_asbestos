#!/bin/bash

# AlexNet -- SigOpt
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_multi \
    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_multi --ignoregit \
    --lr 0.038498 --momentum 0.073146 --weight-decay 0.004074 --decay-lr 20 \
    --multi-run 5

# AlexNet -- Adam optimizer
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_multi_adam \
    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_multi --ignoregit \
    --lr 0.038498 --optimizer-name Adam --decay-lr 20 \
    --multi-run 5

# AlexNet -- SigOpt pre-trained
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_multi_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_multi --ignoregit \
    --lr 0.038498 --momentum 0.073146 --weight-decay 0.004074 --decay-lr 20 \
    --pretrained --multi-run 5