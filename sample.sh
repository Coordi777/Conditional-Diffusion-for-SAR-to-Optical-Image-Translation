#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_P2P_DISABLE=1
export OPENAI_LOGDIR=/dirs/to/save/results
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma False" 
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
SAMPLE_FLAGS="--batch_size 8 --num_samples 100 --timestep_respacing 250"
python scripts/image_sample_realtime.py --model_path /path/to/checkpoints $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

