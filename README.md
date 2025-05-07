# Neural Style Transfer for ECG Images

This project applies neural style transfer techniques to ECG images for various transformations and augmentations. The repository contains scripts, models, and results for experiments conducted on ECG datasets.

## Project Structure

- **src/**: Source code for the project, including transformation scripts and main experiment scripts.
- **run_*.sh**: Shell scripts to execute various experiments and transformations.
- **data.txt**: Link to download data
- **requirements.txt**: Python dependencies required for the project.

## Key Files

- `run_main_merged_augmentation.sh`: Train, evaluate model on both original and nst-transformed data.
- `run_main_nst_augmentation.sh`: Train, evaluate model on only nst-augmented data.
- `run_test_all.sh`: Comparative evaluation of models when only trained on either original or nst-augmented data.
- `run_transform_ab_hb.sh`: Apply NST to abnormal heartbeat images.
- `run_transform_h_MI.sh`: Apply NST to history of myocardial infarction images.
- `run_transform_MI.sh`: Apply NST to myocardial infarction images.
- `run_transform_normal.sh`: Apply NST to normal heartbeat images.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install dependencies using:

  ```bash
  pip install -r requirements.txt
  ```

### Running Experiments (on Athena Cluster)

1. **Without Augmentation**:
   ```bash
   sbatch run_main_no_augmentation.sh
   ```

2. **With Neural Style Transfer Augmentation**:
   ```bash
   sbatch run_main_nst_augmentation.sh
   ```

3. **With Merged Augmentation**:
   ```bash
   sbatch run_main_merged_augmentation.sh
   ```

4. **Testing All Models**:
   ```bash
   sbatch run_test_all.sh
   ```

## Acknowledgments

- Special thanks to the CMSC 636 course for guidance and support.
- Neural Style Transfer techniques inspired by Gatys et al.'s work on artistic style transfer [doi: https://arxiv.org/abs/1508.06576].