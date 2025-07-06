#!/bin/bash
source /parallel_scratch/yc01847/miniconda3/etc/profile.d/conda.sh
conda activate visual-sound

python train.py \
  --hdf5FolderPath /parallel_scratch/yc01847/FAIR-Play/splits/split1 \
  --name mono2binaural \
  --model audioVisual \
  --checkpoints_dir /parallel_scratch/yc01847/2.5D-Visual-Sound/checkpoints \
  --save_epoch_freq 10 \
  --save_latest_freq 5 \
  --display_freq 2 \
  --batchSize 256 \
  --learning_rate_decrease_itr 10 \
  --niter 10 \
  --lr_visual 0.0001 \
  --lr_audio 0.001 \
  --nThreads 4 \
  --gpu_ids 0 \
  --validation_on \
  --validation_freq 5 \
  --validation_batches 2 \
  --tensorboard False \
  |& tee -a mono2binaural.log