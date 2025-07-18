#!/bin/bash


# Run the demo
python demo.py \
  --input_audio_path "/scratch/yc01847/FAIR-Play/binaural_audios/001353.wav" \
  --video_frame_path "/scratch/yc01847/FAIR-Play/frames/001353.mp4" \
  --weights_visual "/scratch/yc01847/2.5D-Visual-Sound/checkpoints/mono2binaural/visual_best.pth" \
  --weights_audio "/scratch/yc01847/2.5D-Visual-Sound/checkpoints/mono2binaural/audio_best.pth" \
  --output_dir_root "/scratch/yc01847/2.5D-Visual-Sound/demo_output" \
  --input_audio_length 10 \
  --hop_size 0.05 \