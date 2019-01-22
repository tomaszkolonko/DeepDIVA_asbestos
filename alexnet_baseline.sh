#!/bin/bash

# AlexNet --  --lr 0.1
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_1 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.1 --optimizer-name Adam --decay-lr 5 --multi-run 5

# AlexNet --  --lr 0.05
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_2 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.05 --optimizer-name Adam --decay-lr 5 --multi-run 5

# AlexNet --  --lr 0.01
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_3 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.01 --optimizer-name Adam --decay-lr 5 --multi-run 5

# AlexNet --  --lr 0.005
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_4 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.005 --optimizer-name Adam --decay-lr 5 --multi-run 5

# AlexNet --  --lr 0.001
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_5 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.001 --optimizer-name Adam --decay-lr 5 --multi-run 5

# AlexNet --  --lr 0.0005
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_6 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.0005 --optimizer-name Adam --decay-lr 5 --multi-run 5

# AlexNet --  --lr 0.0001
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/ASBESTOS/ \
    --model-name alexnet --epochs 20 --experiment-name tz_asbestos_alexnet_7 \
    --output-folder /home/thomas.kolonko/output_asbestos --ignoregit \
    --lr 0.0001 --optimizer-name Adam --decay-lr 5 --multi-run 5
