#!/bin/bash

# densenet121 ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_sigopt \
    --output-folder /home/thomas.kolonko/yolo --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20

