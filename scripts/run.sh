#!/bin/sh

OUTPUTS="/mnt/c/Users/kacem/Workspace/CodeML-2023/Emotion/SourceCode/df/outputs" 
SCRIPTDIR="/mnt/c/Users/kacem/Workspace/CodeML-2023/Emotion/SourceCode/df/scripts"

dataset="emotion"
epochs=200
arch="resnet50"
lr=0.01

suffix="_rand_${lr}"
model_name="${dataset}_${arch}_${epochs}${suffix}"

exp_name="${model_name}"
save_path="${OUTPUTS}/trained_models/${model_name}.pt"

cd $SCRIPTDIR

python ../train_codeml.py \
  --dataset $dataset \
  --save_path $save_path \
  --num_gpus 1 \
  --arch $arch \
  --lr $lr \
  --exp_name $exp_name  \
  --epochs $epochs 

