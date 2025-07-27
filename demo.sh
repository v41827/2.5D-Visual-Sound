#!/bin/bash
source /scratch/yc01847/miniconda3/etc/profile.d/conda.sh
conda activate visual-sound
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run the demo
python demo.py \
  --input_audio_path "/scratch/yc01847/FAIR-Play/binaural_audios/001353.wav" \
  --video_frame_path "/scratch/yc01847/FAIR-Play/frames/001353.mp4" \
  --test_text_path "/scratch/yc01847/FAIR-Play/sampled_output_step10/001353.csv" \
  --hdf5FolderPath "/scratch/yc01847/FAIR-Play/splits/split1" \
  --text_folder_name "sampled_output_step10" \
  --weights_visual "/scratch/yc01847/2.5D-Visual-Sound/checkpoints/AVT_text_frozen/split1/mono2binaural/visual_best.pth" \
  --weights_audio "/scratch/yc01847/2.5D-Visual-Sound/checkpoints/AVT_text_frozen/split1/mono2binaural/audio_best.pth" \
  --freeze_text True \
  --output_dir_root "/scratch/yc01847/2.5D-Visual-Sound/demo_output/AVTsplit1_001353" \
  --input_audio_length 10 \
  --hop_size 0.05 