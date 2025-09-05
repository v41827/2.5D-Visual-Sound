#!/bin/bash


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 train.py \
  --hdf5FolderPath /scratch/yc01847/FAIR-Play/splits/split1 \
  --name mono2binaural \
  --model audioVisual \
  --checkpoints_dir /scratch/yc01847/2.5D-Visual-Sound/checkpoints/split1 \
  --save_epoch_freq 50 \
  --save_latest_freq 100 \
  --display_freq 10 \
  --batchSize 64 \
  --learning_rate_decrease_itr 10 \
  --niter 500 \
  --lr_visual 0.0001 \
  --lr_audio 0.001 \
  --nThreads 5 \
  --gpu_ids 0 \
  --validation_on \
  --validation_freq 100 \
  --validation_batches 50 |& tee -a mono2binaural.log

