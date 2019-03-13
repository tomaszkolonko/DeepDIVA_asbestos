#!/bin/bash

# AlexNet ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# AlexNet PRE-TRAINED ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_sigopt_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained

# Resnet18 ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_resnet18_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# Resnet18 PRE-TRAINED ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_sigopt_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_resnet18_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json