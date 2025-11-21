#!/bin/bash

echo "🔁 Starting PointNetLK inference for STSR Task 2"

python run_inference.py \
  --model_weights pointnetlk_unified_finetuned_norm_outlier_200pretrain_50finetune.pth \
  --input_dir /inputs \
  --output_dir /outputs

echo "✅ Inference completed. Output saved in /outputs"
