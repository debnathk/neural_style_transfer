#!/bin/bash
#SBATCH --job-name=nst_test
#SBATCH --output=./logs/output_test.log   
#SBATCH --error=./logs/error_test.log                          
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G              

echo "Job started at $(date)"  # Job start

# Load cuda
# module load cuda/12.4

# Activate virtual environment (optional)
source .venv/bin/activate

# Run your Python script
mkdir -p results
python test_all.py

echo "All test completed..."