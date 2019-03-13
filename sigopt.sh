#!/bin/bash

# AlexNet ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 5 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# AlexNet ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_alexnet_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 5 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json