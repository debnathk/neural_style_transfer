import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import re
from PIL import Image
from efficientnet_pytorch import EfficientNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the directory where your ECG images are stored
base_dir = './ECG_images_raw'

# Define class names with and without _nst suffix
class_names_nst = ['normal_nst', 'abnormal_hb_nst', 'history_MI_nst', 'MI_nst']
class_names_base = ['normal', 'abnormal_hb', 'history_MI', 'MI']

# Mapping between base classes and their corresponding display names
class_display_names = {
    'normal': 'Normal',
    'abnormal_hb': 'Abnormal HB',
    'history_MI': 'History MI',
    'MI': 'MI'
}

# Function to get paths for all images in each class, including non-_nst folders
def get_image_paths():
    image_paths = {}
    
    # Initialize with empty lists for each class
    for base_class in class_names_base:
        image_paths[base_class] = []
    
    # First, get images from _nst folders
    for class_name in class_names_nst:
        # Extract the base class name (remove _nst suffix)
        base_class = class_name.replace('_nst', '')
        
        class_dir = os.path.join(base_dir, class_name)
        if os.path.exists(class_dir):
            # Get all image files in the class directory
            paths = glob.glob(os.path.join(class_dir, '*.jpg'))
            image_paths[base_class].extend(paths)
            print(f"Found {len(paths)} images for class {class_name}")
        else:
            print(f"Warning: Directory {class_dir} not found")
    
    # Then, get images from non-_nst folders
    for class_name in class_names_base:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.exists(class_dir):
            # Get all image files in the class directory
            paths = glob.glob(os.path.join(class_dir, '*.jpg'))
            image_paths[class_name].extend(paths)
            print(f"Found {len(paths)} additional images for class {class_name}")
        else:
            print(f"Warning: Directory {class_dir} not found")
    
    # Print total counts for each class after merging
    for class_name in class_names_base:
        print(f"Total images for {class_name}: {len(image_paths[class_name])}")
    
    return image_paths

# Create the ECG dataset class
class ECGDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# Define the EfficientNet model
def effnet_model():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)  # 4 classes
    )
    return model

# Define data transformation
data_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705])
])

# Function to create a dataframe from image paths
def create_dataframe(image_paths):
    img_list = []
    img_labels = []
    for class_idx, class_name in enumerate(class_names_base):
        for img_path in image_paths[class_name]:
            img_list.append(img_path)
            img_labels.append(class_idx)
    
    df = pd.DataFrame({'img': img_list, 'label': img_labels})
    return df

# Function to create train and test splits
def create_train_test_split(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Check class distribution in train and test sets
    print("\nTraining set class distribution:")
    print(train_df.label.value_counts())
    print("\nTest set class distribution:")
    print(test_df.label.value_counts())
    
    return train_df, test_df

# Function to evaluate model
def evaluate_model(model, test_loader, model_name="Model"):
    model.eval()
    test_pred = []
    test_actual = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_pred.append(preds.cpu().numpy())
            test_actual.append(labels.numpy())
    
    test_pred = np.concatenate(test_pred)
    test_actual = np.concatenate(test_actual)
    
    # Classification report
    report = classification_report(test_actual, test_pred, target_names=[class_display_names[cls] for cls in class_names_base])
    f1 = f1_score(test_actual, test_pred, average='macro')
    
    # Create confusion matrix
    cm = confusion_matrix(test_actual, test_pred)
    
    # Print results
    print(f"\n==== {model_name} Evaluation ====")
    print(report)
    print(f"Macro F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_display_names[cls] for cls in class_names_base],
                yticklabels=[class_display_names[cls] for cls in class_names_base])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'./results/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    return report, f1, cm

# Function to test all three models on the test set
def test_all_models():
    # Load all image paths
    all_image_paths = get_image_paths()
    
    # Create dataframe
    df = create_dataframe(all_image_paths)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    # Create test dataset and dataloader
    test_dataset = ECGDataset(test_df.img.values, test_df.label.values, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Model file paths
    model_paths = {
        "No Augmentation": "model_no_augmentation.pth",  # Original model with no augmentation
        "NST Augmentation": "model_nst_augmentation.pth",  # NST augmentation only
        # "Merged NST Augmentation": "model_merged_augmentation.pth"  # Merged NST augmentation
    }
    
    # Results storage
    results = {}
    
    for model_name, model_path in model_paths.items():
        try:
            # Initialize model
            model = effnet_model()
            model.to(device)
            
            # Load model weights
            print(f"\nLoading model weights from {model_path}...")
            try:
                model.load_state_dict(torch.load(model_path))
                print(f"Successfully loaded model: {model_name}")
                
                # Evaluate model
                report, f1, cm = evaluate_model(model, test_loader, model_name)
                results[model_name] = {
                    'report': report,
                    'f1': f1,
                    'confusion_matrix': cm
                }
                
            except FileNotFoundError:
                print(f"Model file not found: {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
    
    # Compare F1 scores
    if results:
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        f1_scores = [results[name]['f1'] for name in names]
        
        plt.bar(names, f1_scores, color=['blue', 'green', 'orange'])
        plt.title('Macro F1 Score Comparison')
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        
        # Add values on top of bars
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.ylim(0, max(f1_scores) + 0.1)
        plt.tight_layout()
        plt.savefig('./results/f1_score_comparison.png')
        plt.show()
        
        return results
    else:
        print("No results to compare. Check if model files exist.")
        return None

# Main function to run the testing
def main():
    print("Starting model evaluation...")
    
    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)
    
    # Test all models
    results = test_all_models()
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()