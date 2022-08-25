#!/bin/bash

GPU_ID=0,1

DATASETS=(
          'RAFDB'
          'AffectNet_7'
          'AffectNet_8'
)

NUM_CLASSES=(
            7
            7
            8
)

batch_size=(
            128
            128
            128
)

INIT_LRS=1e-2
ARCH=resnet18

FINETUNE_EPOCHS=100

task_id=1

# CNN20 pretrained on the face verification task
#echo {\"face_verification\": \"0.9942\"} > logs/baseline_face_acc.txt
CUDA_VISIBLE_DEVICES=$GPU_ID python main_affectnet.py \
          --arch $ARCH \
          --dataset ${DATASETS[task_id]} \
  		    --num_classe  ${NUM_CLASSES[task_id]} \
          --lr ${INIT_LRS} \
          --weight_decay 4e-5 \
  	      --batch_size ${batch_size[task_id]} \
  	      --val_batch_size 1 \
          --epochs $FINETUNE_EPOCHS \
          --use_pretrained \
