#!/bin/bash

# FC_4096
# *******

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg16 --epochs 50 --experiment-name tz_asbestos_vgg16_pre \
    --output-folder /home/thomas.kolonko/tz_asbestos_vgg16_pre --ignoregit --lr 0.054173 \
    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --pretrained

python ./template/RunMe.py --runner-class image_classification \
    --dataset-folder /home/thomas.kolonko/FINAL/ \
    --model-name vgg16_bn --epochs 50 --experiment-name tz_asbestos_vgg16_bn \
    --output-folder /home/thomas.kolonko/tz_asbestos_vgg16_bn --ignoregit --lr 0.093533 \
    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20


#
## FC_1024
## *******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_1024_bn --epochs 50 --experiment-name vgg13_fc_1024_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc1024 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_1024_bn --epochs 50 --experiment-name vgg13_fc_1024_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc1024 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained




# FC_512
# *******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_512_bn --epochs 50 --experiment-name vgg13_fc_512_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc512 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_512_bn --epochs 50 --experiment-name vgg13_fc_512_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc512 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained




## FC_256
## ******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_256_bn --epochs 50 --experiment-name vgg13_fc_256_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc256 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_256_bn --epochs 50 --experiment-name vgg13_fc_256_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc256 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained



#
## FC_128
## *******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_128_bn --epochs 50 --experiment-name vgg13_fc_128_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc128 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_128_bn --epochs 50 --experiment-name vgg13_fc_128_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc128 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
#
#
## FC_64
## ******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_64_bn --epochs 50 --experiment-name vgg13_fc_64_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc64 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_64_bn --epochs 50 --experiment-name vgg13_fc_64_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc64 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained




## FC_32
## ******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_32_bn --epochs 50 --experiment-name vgg13_fc_32_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc32 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_32_bn --epochs 50 --experiment-name vgg13_fc_32_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc32 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
#
#
## FC_16
## ******
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_16_bn --epochs 50 --experiment-name vgg13_fc_16_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc16 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_16_bn --epochs 50 --experiment-name vgg13_fc_16_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc16 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained




## FC_8
## *****
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_8_bn --epochs 50 --experiment-name vgg13_fc_8_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc8 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_8_bn --epochs 50 --experiment-name vgg13_fc_8_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc8 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained
#
#
#
#
## FC_4
## *****
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_4_bn --epochs 50 --experiment-name vgg13_fc_4_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc4 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_4_bn --epochs 50 --experiment-name vgg13_fc_4_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc4 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained

#
## FC_2
## *****
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_2_bn --epochs 50 --experiment-name vgg13_fc_2_bn \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc2 --ignoregit --lr 0.054173 \
#    --momentum 0.643504 --weight-decay 0.003223 --decay-lr 20 --multi-run 3
#
#python ./template/RunMe.py --runner-class image_classification \
#    --dataset-folder /home/thomas.kolonko/FINAL/ \
#    --model-name vgg13_fc_2_bn --epochs 50 --experiment-name vgg13_fc_2_bn_pre \
#    --output-folder /home/thomas.kolonko/f_vgg13_fc2 --ignoregit --lr 0.093533 \
#    --momentum 0.041074 --weight-decay 0.009734 --decay-lr 20 --multi-run 3 --pretrained