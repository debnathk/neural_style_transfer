import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
import glob
import re

print(torch.__version__)
print(torchvision.__version__)

# Check cuda
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# device=torch.device('mps')
device=torch.device('cuda')

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

# Load all image paths
all_image_paths = get_image_paths()

# Function to visualize random samples from each class
def visualize_samples(n_samples=3):
    plt.figure(figsize=(15, 12))
    for i, class_name in enumerate(class_names_base):
        if class_name not in all_image_paths or not all_image_paths[class_name]:
            continue
            
        # Select random samples
        samples = random.sample(all_image_paths[class_name], 
                              min(n_samples, len(all_image_paths[class_name])))
        
        for j, img_path in enumerate(samples):
            # Load and display image
            img = Image.open(img_path)
            plt.subplot(len(class_names_base), n_samples, i*n_samples + j + 1)
            plt.imshow(np.array(img))
            plt.title(f"{class_display_names[class_name]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/merged_samples.png')
    plt.show()

# Display class distribution
def show_class_distribution():
    counts = {class_display_names[cls]: len(paths) for cls, paths in all_image_paths.items()}
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values(), color=['green', 'blue', 'orange', 'red'])
    plt.title('Number of Images per Class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    for cls, count in counts.items():
        plt.text(cls, count + 10, str(count), ha='center')
    plt.savefig('./results/merged_class_dist.png')
    plt.show()

# Show random samples from each class
visualize_samples(n_samples=4)

# Show class distribution
show_class_distribution()

# Print total dataset size
total_images = sum(len(paths) for paths in all_image_paths.values())
print(f"Total dataset size: {total_images} images")

# Create a dataset class
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

# Define data transformation
data_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5820, 0.4512, 0.4023], [0.2217, 0.1858, 0.1705])
])

# Create dataframe from image paths
def create_dataframe():
    img_list = []
    img_labels = []
    for class_idx, class_name in enumerate(class_names_base):
        for img_path in all_image_paths[class_name]:
            img_list.append(img_path)
            img_labels.append(class_idx)
    
    df = pd.DataFrame({'img': img_list, 'label': img_labels})
    return df

# Create dataframe
df = create_dataframe()
print(df.head())
print("\nClass distribution:")
print(df.label.value_counts())

# Import EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    
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
except ImportError:
    print("Error: efficientnet_pytorch not found. Please install with: pip install efficientnet_pytorch")

# Define training function
def fit(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs=15, verbose=None):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_sum = 0
        train_acc_sum = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            train_acc_sum += (preds == labels).sum().item() / len(labels)
        
        # Validation phase
        model.eval()
        val_loss_sum = 0
        val_acc_sum = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss_sum += loss.item()
                _, preds = torch.max(outputs, 1)
                val_acc_sum += (preds == labels).sum().item() / len(labels)
        
        # Calculate average metrics
        train_avg_loss = train_loss_sum / len(train_loader)
        val_avg_loss = val_loss_sum / len(val_loader)
        train_avg_acc = train_acc_sum / len(train_loader)
        val_avg_acc = val_acc_sum / len(val_loader)
        
        # Scheduler step
        if scheduler:
            scheduler.step(val_avg_loss)
            
        # Print progress
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Train Acc: {train_avg_acc:.4f}, '
                  f'Val Loss: {val_avg_loss:.4f}, Val Acc: {val_avg_acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save metrics for plotting
        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)
        train_accs.append(train_avg_acc)
        val_accs.append(val_avg_acc)
    
    # Test phase
    test_pred = []
    test_actual = []
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_pred.append(preds.cpu().numpy())
            test_actual.append(labels.numpy())
    
    test_pred = np.concatenate(test_pred)
    test_actual = np.concatenate(test_actual)
    
    # Classification report and F1 score
    report = classification_report(test_actual, test_pred)
    f1 = f1_score(test_actual, test_pred, average='macro')
    
    return [train_losses, val_losses, train_accs, val_accs], [report, f1]

# Train with k-fold cross-validation
def train_kfold(df, n_splits=5, num_epochs=15):
    metrics = []
    results = []
    
    kf = StratifiedKFold(n_splits=n_splits)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df.img, df.label)):
        print(f'-----------fold {fold}--------------')
        
        # Split data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Create datasets
        train_dataset = ECGDataset(train_df.img.values, train_df.label.values, transform=data_transform)
        test_dataset = ECGDataset(test_df.img.values, test_df.label.values, transform=data_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)
        
        # Calculate class weights for balanced loss
        target = [batch[1] for batch in train_loader]
        target_flat = []
        for t in target:
            target_flat.extend(t.tolist())
        target_tensor = torch.tensor(target_flat)
        class_sample_count = torch.unique(target_tensor, return_counts=True)[1]
        weight = torch.true_divide(1, class_sample_count)
        
        # Initialize model
        model = effnet_model()
        model = model.to(device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=weight.to(device))
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Train and evaluate
        fold_metrics, fold_results = fit(
            train_loader, val_loader, val_loader, 
            model, criterion, optimizer, scheduler,
            num_epochs=num_epochs, verbose=True
        )
        
        metrics.append(fold_metrics)
        results.append(fold_results)
        
    return metrics, results, model

# Run training
metrics, results, model = train_kfold(df, n_splits=5, num_epochs=15)

# Plot results
def plot_metrics(metrics):
    # Loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    for i in range(5):
        plt.plot(metrics[i][1], label=f'Fold{i+1}', linestyle='-.', linewidth=1.5)
    
    # Average validation loss across folds
    avg_val_loss = np.mean([metrics[i][1] for i in range(5)], axis=0)
    plt.plot(avg_val_loss, label='Average', linewidth=2, color='black')
    
    plt.title('Validation Loss for Each Fold')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    
    for i in range(5):
        plt.plot(metrics[i][3], label=f'Fold{i+1}', linestyle='-.', linewidth=1.5)
    
    # Average validation accuracy across folds
    avg_val_acc = np.mean([metrics[i][3] for i in range(5)], axis=0)
    plt.plot(avg_val_acc, label='Average', linewidth=2, color='black')
    
    plt.title('Validation Accuracy for Each Fold')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/merged_val_loss_acc.png')
    plt.show()
    
    # Average training vs validation metrics
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    avg_train_loss = np.mean([metrics[i][0] for i in range(5)], axis=0)
    avg_val_loss = np.mean([metrics[i][1] for i in range(5)], axis=0)
    
    plt.plot(avg_train_loss, label='Train Average')
    plt.plot(avg_val_loss, label='Val Average')
    plt.title('Averaged Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    avg_train_acc = np.mean([metrics[i][2] for i in range(5)], axis=0)
    avg_val_acc = np.mean([metrics[i][3] for i in range(5)], axis=0)
    
    plt.plot(avg_train_acc, label='Train Average')
    plt.plot(avg_val_acc, label='Val Average')
    plt.title('Averaged Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/merged_train_val_loss_accuracy.png')
    plt.show()

# Display classification reports
def display_results(results):
    print("Classification Reports for Each Fold:")
    for i, (report, f1) in enumerate(results):
        print(f"\nFold {i}:")
        print(report)
        print(f"Macro F1-Score: {f1:.4f}")
    
    # Class mappings
    class_mapping = {
        '0': class_display_names['normal'],
        '1': class_display_names['abnormal_hb'],
        '2': class_display_names['history_MI'],
        '3': class_display_names['MI'],
        'macro avg': 'macro',
        'weighted avg': 'weighted'
    }

    # Initialize dictionary to store all metrics
    all_metrics = {name: {'precision': [], 'recall': [], 'f1-score': [], 'support': []} 
                for name in class_mapping.values()}

    # Extract metrics from each report
    for report_text, _ in results:
        for line in report_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains metrics
            parts = re.split(r'\s+', line)
            
            # Handle class metrics and average metrics
            for key in class_mapping:
                if line.startswith(key) or (key.isdigit() and parts[0] == key):
                    if key.isdigit():  # Class metrics
                        class_name = class_mapping[key]
                        values = parts[1:5]
                    else:  # Avg metrics
                        class_name = class_mapping[key]
                        values = parts[-4:]
                    
                    # Extract values
                    try:
                        precision, recall, f1, support = map(float, values)
                        all_metrics[class_name]['precision'].append(precision)
                        all_metrics[class_name]['recall'].append(recall)
                        all_metrics[class_name]['f1-score'].append(f1)
                        all_metrics[class_name]['support'].append(support)
                    except (ValueError, IndexError):
                        continue

    # Calculate averages and print
    print(f"{'':15} {'precision':10} {'recall':8} {'f1-score':10} {'support':8}")
    for class_name in list(class_display_names.values()) + ['macro', 'weighted']:
        if class_name in all_metrics:
            metrics = all_metrics[class_name]
            avg_precision = np.mean(metrics['precision'])
            avg_recall = np.mean(metrics['recall'])
            avg_f1 = np.mean(metrics['f1-score'])
            avg_support = np.mean(metrics['support'])
            
            print(f"{class_name:15} {avg_precision:.3f}    {avg_recall:.3f}     {avg_f1:.3f}    {avg_support:.1f}")

# Plot metrics and display results
plot_metrics(metrics)
display_results(results)

# Save the merged model
torch.save(model.state_dict(), 'model_merged_augmentation.pth')