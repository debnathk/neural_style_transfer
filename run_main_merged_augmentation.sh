#!/bin/bash
#SBATCH --job-name=nst_merged_aug
#SBATCH --output=./logs/output_nst_merged_aug.log   
#SBATCH --error=./logs/error_nst_merged_aug.log                          
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G              

start=$(date +%s)
echo "Job started at $(date)"  # Job start

# Load cuda
# module load cuda/12.4

# Activate virtual environment (optional)
source .venv/bin/activate

# Run your Python script
mkdir -p results
python main_merged_augmentation.py

end=$(data +%s)
elapsed_time=$((end - start))
echo "Time elapsed $elapsed_time seconds"