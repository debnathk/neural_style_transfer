# Neural Style Transfer for ECG Images

This project applies neural style transfer techniques to ECG images for various transformations and augmentations. The repository contains scripts, models, and results for experiments conducted on ECG datasets.

## Project Structure

- **ECG_images_raw/**: Contains raw ECG images categorized into different folders such as `abnormal_hb`, `normal`, `MI`, etc.
- **logs/**: Stores log files for errors and outputs during the execution of scripts.
- **model_weights/**: Pre-trained model weights used for different experiments.
- **results/**: Contains visualizations, metrics, and comparison results from the experiments.
- **src/**: Source code for the project, including transformation scripts and main experiment scripts.
- **run_*.sh**: Shell scripts to execute various experiments and transformations.
- **requirements.txt**: Python dependencies required for the project.

## Key Files

- `main_merged_augmentation.py`: Script for training and testing with merged augmentations.
- `main_no_augmentation.py`: Script for training and testing without augmentations.
- `main_nst_augmentation.py`: Script for training and testing with neural style transfer augmentations.
- `test_all.py`: Script to test all models and generate results.
- `ecg_transformation_nst.py`: Script for applying neural style transfer to ECG images.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install dependencies using:

  ```bash
  pip install -r requirements.txt
  ```

### Running Experiments

1. **Without Augmentation**:
   ```bash
   python src/main_no_augmentation.py
   ```

2. **With Neural Style Transfer Augmentation**:
   ```bash
   python src/main_nst_augmentation.py
   ```

3. **With Merged Augmentation**:
   ```bash
   python src/main_merged_augmentation.py
   ```

4. **Testing All Models**:
   ```bash
   python src/test_all.py
   ```

### Logs and Results

- Logs are stored in the `logs/` directory.
- Results, including metrics and visualizations, are stored in the `results/` directory.

## Notes

- The `.gitignore` file excludes large directories like `ECG_images_raw`, `logs`, `model_weights`, and `results` to keep the repository lightweight.
- Ensure the `ECG_images_raw` directory is populated with the required images before running the scripts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Special thanks to the CMSC 636 course for guidance and support.
- Neural Style Transfer techniques inspired by Gatys et al.'s work on artistic style transfer.