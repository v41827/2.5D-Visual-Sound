python train.py \
  --hdf5FolderPath /YOUR_CODE_PATH/2.5d_visual_sound/hdf5/ \       # Path to HDF5 dataset
  --name mono2binaural \                                            # Name of the experiment
  --model audioVisual \                                             # Model type
  --checkpoints_dir /YOUR_CHECKPOINT_PATH/ \                        # Directory to save checkpoints

  # Saving and display frequencies
  --save_epoch_freq 10 \                                            # Save checkpoint every 50 epochs
  --save_latest_freq 5 \                                          # Save latest model every 100 iterations
  --display_freq 2 \                                               # Display training progress every 10 iterations

  # Training settings
  --batchSize 256 \                                                 # Batch size
  --learning_rate_decrease_itr 10 \                                 # Interval to decrease learning rate
  --niter 10 \                                                    # Number of iterations
  --lr_visual 0.0001 \                                              # Learning rate for visual branch
  --lr_audio 0.001 \                                                # Learning rate for audio branch

  # Hardware settings
  --nThreads 4 \                                                   # Number of data loading threads
  --gpu_ids 0 \                                       # Use GPUs 0–7

  # Validation settings
  --validation_on \                                                 # Enable validation
  --validation_freq  5 \                                           # Run validation every 100 iterations
  --validation_batches 2 \                                         # Number of validation batches

  # Logging
  --tensorboard False \                                              # Enable TensorBoard logging

  # Save both stdout and stderr to log file
  |& tee -a mono2binaural.log