#!/bin/bash

## AlexNet ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_sigopt \
#    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json
#
## AlexNet PRE-TRAINED ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name alexnet --epochs 50 --experiment-name tz_asbestos_alexnet_sigopt_pre \
#    --output-folder /home/thomas.kolonko/output_asbestos_alexnet_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained
#
## Resnet18 ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_sigopt \
#    --output-folder /home/thomas.kolonko/output_asbestos_resnet18_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json

## Resnet18 PRE-TRAINED ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name resnet18 --epochs 50 --experiment-name tz_asbestos_resnet18_sigopt_pre \
#    --output-folder /home/thomas.kolonko/output_asbestos_resnet18_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained
#
## Resnet34 ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name resnet34 --epochs 50 --experiment-name tz_asbestos_resnet34_sigopt \
#    --output-folder /home/thomas.kolonko/output_asbestos_resnet34_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json
#
## Resnet34 PRE-TRAINED ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name resnet34 --epochs 50 --experiment-name tz_asbestos_resnet34_sigopt_pre \
#    --output-folder /home/thomas.kolonko/output_asbestos_resnet34_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained
#
## densenet121 ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_sigopt \
#    --output-folder /home/thomas.kolonko/output_asbestos_densenet121_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json
#
## densenet121 PRE-TRAINED ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name densenet121 --epochs 50 --experiment-name tz_asbestos_densenet121_sigopt_pre \
#    --output-folder /home/thomas.kolonko/output_asbestos_densenet121_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained

## densenet169 ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name densenet169 --epochs 50 --experiment-name tz_asbestos_densenet169_sigopt \
#    --output-folder /home/thomas.kolonko/output_asbestos_densenet169_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json
#
## densenet169 PRE-TRAINED ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name densenet169 --epochs 50 --experiment-name tz_asbestos_densenet169_sigopt_pre \
#    --output-folder /home/thomas.kolonko/output_asbestos_densenet169_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained


# inception_vgg11 ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg11 --epochs 50 --experiment-name tz_asbestos_vgg11_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg11_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# inception_vgg11_bn PRE-TRAINED ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg11_bn --epochs 50 --experiment-name tz_asbestos_vgg11_bn_sigopt_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg11_bn_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained

# inception_vgg13 ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13 --epochs 50 --experiment-name tz_asbestos_vgg13_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg13_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# inception_vgg13_bn PRE-TRAINED ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg13_bn --epochs 50 --experiment-name tz_asbestos_vgg13_bn_sigopt_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg13_bn_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained

# inception_vgg16 ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg16 --epochs 50 --experiment-name tz_asbestos_vgg16_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg16_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# inception_vgg16_bn PRE-TRAINED ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg16_bn --epochs 50 --experiment-name tz_asbestos_vgg16_bn_sigopt_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg16_bn_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained

# inception_vgg19 ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg19 --epochs 50 --experiment-name tz_asbestos_vgg19_sigopt \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg19_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json

# inception_vgg19_bn PRE-TRAINED ----------------------------------------
python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg19_bn --epochs 50 --experiment-name tz_asbestos_vgg11_bn_sigopt_pre \
    --output-folder /home/thomas.kolonko/output_asbestos_vgg19_bn_sigopt --ignoregit --lr 0.01 \
    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained

## inception_v3 ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_v3_sigopt \
#    --output-folder /home/thomas.kolonko/output_asbestos_inception_v3_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json
#
## inception_v3 PRE-TRAINED ----------------------------------------
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name inception_v3 --epochs 50 --experiment-name tz_asbestos_inception_v3_sigopt_pre \
#    --output-folder /home/thomas.kolonko/output_asbestos_inception_v3_sigopt --ignoregit --lr 0.01 \
#    --momentum 0.9 --decay-lr 20 --sig-opt-token CEUJGDONOZMDYVYJLRWBJUGTQZKCLTFFPFGGKQDUDBZDHZCL \
#    --sig-opt-runs 30 --sig-opt util/sigopt.json --pretrained