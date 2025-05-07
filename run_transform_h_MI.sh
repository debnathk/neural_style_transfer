#!/bin/bash
#SBATCH --job-name=nst_h_MI
#SBATCH --output=./logs/nst_output_h_MI.log   
#SBATCH --error=./logs/nst_error_h_MI.log                          
#SBATCH --gres=gpu:1                 
#SBATCH --partition=gpu
#SBATCH --mem=32G              

start = $(date +%s)
echo "Job started at $(date)"  # Job start

# Load cuda
module load cuda/12.4

# Activate virtual environment (optional)
source .venv/bin/activate

# Run your Python script
python ecg_transformation_nst.py --ab_type history_MI

end = $(data +%s)
echo "Time elapsed $((end - start)) seconds"