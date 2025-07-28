import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import warnings
import torchvision.transforms as transforms
import random
from torchvision.models import efficientnet_b0, efficientnet_b2
warnings.filterwarnings('ignore')

class EEGSpectrogramDataset(Dataset):
    def init(self, csv_file, eeg_dir, sample_rate=200, duration=50, transform=None, limit=None, augment=False):
        self.df = pd.read_csv(csv_file)
        if limit is not None:
            self.df = self.df.iloc[:limit]
        self.eeg_dir = eeg_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.augment = augment

        # Spectrogram parameters
        self.nperseg = 256  # window length
        self.noverlap = 128  # overlap length
        self.nfft = 256     # FFT points

    def len(self):
        return len(self.df)

    def create_spectrogram(self, eeg_data):
        """Create spectrograms for each channel"""
        spectrograms = []
        for channel in range(eeg_data.shape[0]):
            # EEG data augmentation
            f, t, Sxx = signal.spectrogram(
                eeg_data[channel], 
                fs=self.sample_rate,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft
            )

            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            Sxx_db = (Sxx_db - np.mean(Sxx_db)) / (np.std(Sxx_db) + 1e-8)
            spectrograms.append(Sxx_db)

        return np.array(spectrograms)  # shape: (channels, freq_bins, time_bins)

    def augment_eeg_data(self, eeg_data):
        """EEG data augmentation"""
        augmented_data = eeg_data.copy()

        # 1. Time shift
        if random.random() > 0.5:
            shift = random.randint(-int(0.1 * eeg_data.shape[1]), int(0.1 * eeg_data.shape[1]))
            if shift > 0:
                augmented_data = np.concatenate([augmented_data[:, shift:], augmented_data[:, :shift]], axis=1)
            elif shift < 0:
                augmented_data = np.concatenate([augmented_data[:, shift:], augmented_data[:, :shift]], axis=1)

        # 2. Noise addition
        if random.random() > 0.5:
            noise_factor = random.uniform(0.01, 0.05)
            noise = np.random.randn(*augmented_data.shape) * noise_factor * np.std(augmented_data)
            augmented_data = augmented_data + noise

        # 3. Amplitude scaling
        if random.random() > 0.5:
            scale_factor = random.uniform(0.8, 1.2)
            augmented_data = augmented_data * scale_factor

        # 4. Time warping
        if random.random() > 0.7:
            stretch_factor = random.uniform(0.95, 1.05)
            original_length = augmented_data.shape[1]
            new_length = int(original_length * stretch_factor)
            if new_length != original_length:
                # Simple linear interpolation to adjust the length
                indices = np.linspace(0, original_length - 1, new_length)
                augmented_data = np.array([np.interp(indices, range(original_length), channel) 
                                         for channel in augmented_data])
                # Adjust back to original length
                if new_length > original_length:
                    augmented_data = augmented_data[:, :original_length]
                else:
                    pad_length = original_length - new_length
                    padding = np.zeros((augmented_data.shape[0], pad_length))
                    augmented_data = np.concatenate([augmented_data, padding], axis=1)

        return augmented_data

    def augment_spectrogram(self, spectrogram):
        """Spectrogram augmentation"""
        augmented_spec = spectrogram.copy()

        # 1. Frequency masking
        if random.random() > 0.5:
            freq_mask_param = int(0.1 * augmented_spec.shape[1])  # 10% of frequency bins
            f0 = random.randint(0, augmented_spec.shape[1] - freq_mask_param)
            augmented_spec[:, f0:f0+freq_mask_param, :] = 0

        # 2. Time masking 
        if random.random() > 0.5:
            time_mask_param = int(0.1 * augmented_spec.shape[2])  # 10% of time bins
            t0 = random.randint(0, augmented_spec.shape[2] - time_mask_param)
            augmented_spec[:, :, t0:t0+time_mask_param] = 0
        # 3. Amplitude Scaling
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            augmented_spec = augmented_spec * scale_factor

        return augmented_spec

    def getitem(self, idx):
        max_retries = 5
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.df)
                row = self.df.iloc[current_idx]

                # Read EEG signal file
                eeg_id = row["eeg_id"]
                eeg_path = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
                eeg_data = pd.read_parquet(eeg_path)

                # Calculating fragment positions
                offset_sec = int(row["eeg_label_offset_seconds"])
                start = offset_sec * self.sample_rate
                end = start + self.duration * self.sample_rate

                if end > len(eeg_data):
                    end = len(eeg_data)
                    start = max(0, end - self.duration * self.sample_rate)

                eeg_segment = eeg_data.iloc[start:end].values
                if len(eeg_segment) < self.duration * self.sample_rate:
                    pad_length = self.duration * self.sample_rate - len(eeg_segment)
                    padding = np.zeros((pad_length, eeg_segment.shape[1]))
                    eeg_segment = np.vstack([eeg_segment, padding])

                eeg_segment = eeg_segment.T  

                # Cleaning outliers
                eeg_segment = np.nan_to_num(eeg_segment, nan=0.0, posinf=0.0, neginf=0.0)

                # Improved standardization
                for i in range(eeg_segment.shape[0]):
                    channel_data = eeg_segment[i]
                    # Use a more robust normalization method
                    percentile_01 = np.percentile(channel_data, 1)
                    percentile_99 = np.percentile(channel_data, 99)

                    # Clipping extreme values
                    channel_data = np.clip(channel_data, percentile_01, percentile_99)

                    # standardization
                    mean_val = np.mean(channel_data)
                    std_val = np.std(channel_data)
                    if std_val > 1e-8:
                        eeg_segment[i] = (channel_data - mean_val) / std_val
                    else:
                        eeg_segment[i] = channel_data - mean_val

                # Data Augmentation
                if self.augment:
                    eeg_segment = self.augment_eeg_data(eeg_segment)

                # Generate Spectrum Plot
                spectrogram = self.create_spectrogram(eeg_segment)

                # Spectrogram Enhancement
                if self.augment:
                    spectrogram = self.augment_spectrogram(spectrogram)

                # Generate soft label vector
                votes = np.array([
                    row['seizure_vote'],
                    row['lpd_vote'],
                    row['gpd_vote'],
                    row['lrda_vote'],
                    row['grda_vote'],
                    row['other_vote']
                ], dtype=np.float32)
                votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)
                vote_sum = votes.sum()
                if vote_sum <= 0:
                    votes = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
                    vote_sum = 1.0

                label = votes / vote_sum

                # Final Check
                if np.isnan(eeg_segment).any() or np.isinf(eeg_segment).any():
                    continue
                if np.isnan(spectrogram).any() or np.isinf(spectrogram).any():
                    continue
                if np.isnan(label).any() or np.isinf(label).any():
                    continue

                return (torch.tensor(eeg_segment, dtype=torch.float32), 
                        torch.tensor(spectrogram, dtype=torch.float32), 
                        torch.tensor(label, dtype=torch.float32))

            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Unable to load sample {idx}，Use the default sample")
                    default_eeg = np.random.randn(20, self.duration * self.sample_rate) * 0.1
                    default_spec = np.random.randn(20, 129, 391) * 0.1
                    default_label = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
                    return (torch.tensor(default_eeg, dtype=torch.float32), 
                            torch.tensor(default_spec, dtype=torch.float32),
                            torch.tensor(default_label, dtype=torch.float32))
                continue

        return self.getitem(0)
class EfficientNetBackbone(nn.Module):
    """Feature Extractor Based on EfficientNet"""
    def init(self, model_name='efficientnet_b2', pretrained=True):
        super(EfficientNetBackbone, self).init()

        if model_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif model_name == 'efficientnet_b2':
            self.backbone = efficientnet_b2(pretrained=pretrained)
            self.feature_dim = 1408

        # Remove the classifier layer
        self.backbone.classifier = nn.Identity()

        # Modify the first layer to accept 20 EEG channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            20,  # 20 EEG channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

    def forward(self, x):
        return self.backbone(x)
class MultiModalEfficientNet(nn.Module):
    def init(self, eeg_channels=20, spec_channels=20, num_classes=6, backbone='efficientnet_b2'):
        super(MultiModalEfficientNet, self).init()

        # EEG branch - Enhanced 1D CNN
        self.eeg_conv1 = nn.Conv1d(eeg_channels, 128, kernel_size=15, padding=7)
        self.eeg_bn1 = nn.BatchNorm1d(128)
        self.eeg_pool1 = nn.MaxPool1d(4)
        self.eeg_dropout1 = nn.Dropout(0.2)

        self.eeg_conv2 = nn.Conv1d(128, 256, kernel_size=9, padding=4)
        self.eeg_bn2 = nn.BatchNorm1d(256)
        self.eeg_pool2 = nn.MaxPool1d(4)
        self.eeg_dropout2 = nn.Dropout(0.3)
        self.eeg_conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.eeg_bn3 = nn.BatchNorm1d(512)
        self.eeg_pool3 = nn.MaxPool1d(4)
        self.eeg_dropout3 = nn.Dropout(0.3)

        # Spectrogram branch - EfficientNet
        self.spec_backbone = EfficientNetBackbone(backbone, pretrained=True)

        # Global average pooling
        self.eeg_gap = nn.AdaptiveAvgPool1d(1)
        self.spec_gap = nn.AdaptiveAvgPool2d(1)

        # Attention mechanisms
        self.eeg_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        self.spec_attention = nn.Sequential(
            nn.Linear(self.spec_backbone.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.spec_backbone.feature_dim),
            nn.Sigmoid()
        )

        # Fusion layers
        fusion_dim = 512 + self.spec_backbone.feature_dim
        self.fusion_fc1 = nn.Linear(fusion_dim, 1024)
        self.fusion_bn1 = nn.BatchNorm1d(1024)
        self.fusion_dropout1 = nn.Dropout(0.5)

        self.fusion_fc2 = nn.Linear(1024, 512)
        self.fusion_bn2 = nn.BatchNorm1d(512)
        self.fusion_dropout2 = nn.Dropout(0.4)

        self.fusion_fc3 = nn.Linear(512, 256)
        self.fusion_bn3 = nn.BatchNorm1d(256)
        self.fusion_dropout3 = nn.Dropout(0.3)

        self.output_fc = nn.Linear(256, num_classes)
    def forward(self, eeg_data, spec_data):
        # EEG branch
        eeg_x = F.relu(self.eeg_bn1(self.eeg_conv1(eeg_data)))
        eeg_x = self.eeg_pool1(eeg_x)
        eeg_x = self.eeg_dropout1(eeg_x)

        eeg_x = F.relu(self.eeg_bn2(self.eeg_conv2(eeg_x)))
        eeg_x = self.eeg_pool2(eeg_x)
        eeg_x = self.eeg_dropout2(eeg_x)

        eeg_x = F.relu(self.eeg_bn3(self.eeg_conv3(eeg_x)))
        eeg_x = self.eeg_pool3(eeg_x)
        eeg_x = self.eeg_dropout3(eeg_x)

        # Global Average Pooling
        eeg_x = self.eeg_gap(eeg_x).squeeze(-1)  # [B, 512]

        # EEG attention
        eeg_att = self.eeg_attention(eeg_x)
        eeg_x = eeg_x * eeg_att

        # Spectrogram branch
        spec_x = self.spec_backbone(spec_data)  # [B, feature_dim]

        # Spectrogram attention
        spec_att = self.spec_attention(spec_x)
        spec_x = spec_x * spec_att

        # Fusion
        fused_x = torch.cat([eeg_x, spec_x], dim=1)

        fused_x = F.relu(self.fusion_bn1(self.fusion_fc1(fused_x)))
        fused_x = self.fusion_dropout1(fused_x)

        fused_x = F.relu(self.fusion_bn2(self.fusion_fc2(fused_x)))
        fused_x = self.fusion_dropout2(fused_x)

        fused_x = F.relu(self.fusion_bn3(self.fusion_fc3(fused_x)))
        fused_x = self.fusion_dropout3(fused_x)

        output = self.output_fc(fused_x)

        return output
      
# Loss functions
class FocalLoss(nn.Module):
    def init(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).init()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# KL divergence loss function, more suitable for processing soft labels
def kl_divergence_loss(predictions, targets):
    """Calculate KL divergence loss"""
    log_probs = F.log_softmax(predictions, dim=1)
    kl_loss = F.kl_div(log_probs, targets, reduction='batchmean')
    return kl_loss
# Label smoothed cross entropy loss
def label_smoothing_loss(predictions, targets, smoothing=0.1):
    """Label smoothed cross entropy loss"""
    log_probs = F.log_softmax(predictions, dim=1)
    smooth_targets = targets * (1 - smoothing) + smoothing / targets.size(1)
    loss = -torch.sum(smooth_targets * log_probs, dim=1).mean()
    return loss
# Improved combined loss function
def advanced_combined_loss(predictions, targets, alpha=0.5, beta=0.3, gamma=0.2):
    """Combining KL divergence, label smoothing loss and focal loss"""
    kl_loss = kl_divergence_loss(predictions, targets)
    smooth_loss = label_smoothing_loss(predictions, targets, smoothing=0.1)

    # Convert soft labels to hard labels for focal loss
    hard_targets = torch.argmax(targets, dim=1)
    focal_loss_fn = FocalLoss(alpha=1, gamma=2)
    focal_loss = focal_loss_fn(predictions, hard_targets)

    return alpha * kl_loss + beta * smooth_loss + gamma * focal_loss
# Dataset Splitting
print("Preparing the dataset...")
full_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
print(f"Full dataset size: {len(full_df)}")
# Dataset Splitting
train_df, val_df = train_test_split(full_df, test_size=0.15, random_state=42, 
                                    stratify=full_df['expert_consensus'])
# Save a temporary CSV file
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
# Create a dataset (training set uses data augmentation)
train_dataset = EEGSpectrogramDataset(
    csv_file="train_split.csv",
    eeg_dir="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs",
    augment=True  # Enable data augmentation on the training set
)
val_dataset = EEGSpectrogramDataset(
    csv_file="val_split.csv",
    eeg_dir="/kaggle/input/hms-harmful-brain-activity-classification/train_eegs",
    augment=False  # No data augmentation is used on the validation set
)
# Creating a DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=12,  
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use equipment: {device}")
# Model, Optimizer, Scheduler
model = MultiModalEfficientNet(backbone='efficientnet_b2').to(device)
# Use more advanced optimizer settings
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, 
                       betas=(0.9, 0.999), eps=1e-8)
# Improved learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=2e-4,
    steps_per_epoch=len(train_loader),
    epochs=50,
    pct_start=0.1,
    anneal_strategy='cos'
)
# Gradient Clipping
def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_gradnorm(model.parameters(), max_norm)
# Model evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for eeg_batch, spec_batch, y_batch in data_loader:
            eeg_batch = eeg_batch.to(device)
            spec_batch = spec_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(eeg_batch, spec_batch)
            loss = advanced_combined_loss(logits, y_batch)
            total_loss += loss.item()
            num_batches += 1

            # Prediction results
            predictions = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
            true_classes = torch.argmax(y_batch, dim=1)

            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(true_classes.cpu().numpy())

    avg_loss = total_loss / num_batches
    return np.array(all_predictions), np.array(all_labels), avg_loss
# Early Stopping Mechanism
class EarlyStopping:
    def init(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    def call(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
      
# Training loop
print("Start training...")
num_epochs = 50 
best_val_loss = float('inf')
train_losses = []
val_losses = []
learning_rates = []

# Initializing Early Stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    valid_batches = 0

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)

    for batch_idx, (eeg_batch, spec_batch, y_batch) in enumerate(train_loader):
        eeg_batch = eeg_batch.to(device)
        spec_batch = spec_batch.to(device)
        y_batch = y_batch.to(device)

        # Check input for outliers
        if torch.isnan(eeg_batch).any() or torch.isinf(eeg_batch).any():
            continue
        if torch.isnan(spec_batch).any() or torch.isinf(spec_batch).any():
            continue

        optimizer.zero_grad()
        logits = model(eeg_batch, spec_batch)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            continue

        loss = advanced_combined_loss(logits, y_batch)

        if torch.isnan(loss):
            continue

        loss.backward()
        clip_gradients(model, max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Update the learning rate after each batch

        total_loss += loss.item()
        num_batches += 1
        valid_batches += 1

        if batch_idx % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

    # Record learning rate
    learning_rates.append(scheduler.get_last_lr()[0])

    # Calculate the average training loss
    if valid_batches > 0:
        avg_train_loss = total_loss / valid_batches
        train_losses.append(avg_train_loss)
        print(f"  Training loss: {avg_train_loss:.4f}")
    else:
        continue

    # Verification Phase
    print("  Verifying...")
    predictions, true_labels, avg_val_loss = evaluate_model(model, val_loader, device)
    val_losses.append(avg_val_loss)

    val_accuracy = (predictions == true_labels).mean()
    print(f"  Validation loss: {avg_val_loss:.4f}, Verification accuracy: {val_accuracy:.4f}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_efficientnet_eeg_model.pth')
        print(f"  Save the best model (Validation loss: {avg_val_loss:.4f})")

    # Early stop check
    if early_stopping(avg_val_loss, model):
        print(f"  Early stop at epoch {epoch+1}")
        break
print("\nTraining Completed！")
# Final Assessment
model.load_state_dict(torch.load('best_efficientnet_eeg_model.pth'))
print("The best model has been loaded for final evaluation")
predictions, true_labels, final_val_loss = evaluate_model(model, val_loader, device)
print(f"Final validation loss: {final_val_loss:.4f}")
# Category Name
class_names = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
# Generate classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_names))
# Calculate accuracy
accuracy = (predictions == true_labels).mean()
print(f"\nOverall accuracy: {accuracy:.4f}")
# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True label')
plt.show()
# Save the final model
torch.save(model.state_dict(), 'final_multimodal_eeg_model.pth')
print("Final model saved")
# Cleaning up temporary files
os.remove('train_split.csv')
os.remove('val_split.csv')
print("Temporary files cleaned")
