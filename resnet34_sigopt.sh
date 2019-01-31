#!/bin/bash

# AlexNet ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 5 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json