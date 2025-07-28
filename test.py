import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import warnings
from torchvision.models import efficientnet_b0, efficientnet_b2
warnings.filterwarnings('ignore')

# Test dataset class (matching data processing during training)
class EEGTestSpectrogramDataset(Dataset):
    def __init__(self, csv_file, eeg_dir, sample_rate=200, duration=50, transform=None):
        self.df = pd.read_csv(csv_file)
        self.eeg_dir = eeg_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        
        # Spectrogram parameters (same as training)
        self.nperseg = 256  # Window length
        self.noverlap = 128  # Overlap length
        self.nfft = 256     # FFT points
    
    def __len__(self):
        return len(self.df)
    
    def create_spectrogram(self, eeg_data):
        """Create a spectrogram for each channel"""
        spectrograms = []
        for channel in range(eeg_data.shape[0]):
            # Compute the spectrogram
            f, t, Sxx = signal.spectrogram(
                eeg_data[channel], 
                fs=self.sample_rate,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft
            )
            
            # Convert to dB and normalize
            Sxx_db = 10 * np.log10(Sxx + 1e-12)
            Sxx_db = (Sxx_db - np.mean(Sxx_db)) / (np.std(Sxx_db) + 1e-8)
            
            spectrograms.append(Sxx_db)
        
        return np.array(spectrograms)  # shape: (channels, freq_bins, time_bins)
    
    def __getitem__(self, idx):
        max_retries = 3
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.df)
                row = self.df.iloc[current_idx]
                
                # Read EEG signal file
                eeg_id = row["eeg_id"]
                eeg_path = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
                eeg_data = pd.read_parquet(eeg_path)
                
                # For test data, use the entire file or the first 50 seconds of data.
                target_length = self.duration * self.sample_rate
                if len(eeg_data) < target_length:
                    # If the data is not long enough, fill it
                    pad_length = target_length - len(eeg_data)
                    padding = np.zeros((pad_length, eeg_data.shape[1]))
                    eeg_segment = np.vstack([eeg_data.values, padding])
                else:
                    # If the data is long enough, take the first target_length samples
                    eeg_segment = eeg_data.iloc[:target_length].values
                
                eeg_segment = eeg_segment.T  
                
                # Cleaning outliers
                eeg_segment = np.nan_to_num(eeg_segment, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Improved normalization (consistent with training)
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

                # Generate Spectrum Plot
                spectrogram = self.create_spectrogram(eeg_segment)
                
                # Final Check
                if np.isnan(eeg_segment).any() or np.isinf(eeg_segment).any():
                    continue
                if np.isnan(spectrogram).any() or np.isinf(spectrogram).any():
                    continue
                
                return (torch.tensor(eeg_segment, dtype=torch.float32), 
                        torch.tensor(spectrogram, dtype=torch.float32), 
                        eeg_id)
                        
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Unable to load sample {idx}, using default sample")
                    default_eeg = np.random.randn(20, self.duration * self.sample_rate) * 0.1
                    default_spec = np.random.randn(20, 129, 391) * 0.1
                    return (torch.tensor(default_eeg, dtype=torch.float32), 
                            torch.tensor(default_spec, dtype=torch.float32),
                            f"default_{idx}")
                continue
        
        return self.__getitem__(0)

class EfficientNetBackbone(nn.Module):
    """EfficientNet-based feature extractor - Offline version"""
    def __init__(self, model_name='efficientnet_b2', pretrained=False):
        super(EfficientNetBackbone, self).__init__()
        
        if model_name == 'efficientnet_b0':
            # Do not use pretrained=True to avoid downloading from the Internet
            self.backbone = efficientnet_b0(pretrained=False)
            self.feature_dim = 1280
        elif model_name == 'efficientnet_b2':
            self.backbone = efficientnet_b2(pretrained=False)
            self.feature_dim = 1408
        
        # Remove the classification layer
        self.backbone.classifier = nn.Identity()
        
        # Modify the first layer to adapt to different input channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            20,  
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
    
    def forward(self, x):
        return self.backbone(x)
# Multimodal EfficientNet model (exactly the same as during training)
class MultiModalEfficientNet(nn.Module):
    def __init__(self, eeg_channels=20, spec_channels=20, num_classes=6, backbone='efficientnet_b2'):
        super(MultiModalEfficientNet, self).__init__()
        
        # EEG branch - improved 1D CNN
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
        
        # Global Average Pooling
        self.eeg_gap = nn.AdaptiveAvgPool1d(1)
        self.spec_gap = nn.AdaptiveAvgPool2d(1)
        
        # Attention Mechanism
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
        
        # Fusion Layer
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
        # EEG branch processing
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
        
        # Spectrogram Branch processing
        spec_x = self.spec_backbone(spec_data)  # [B, feature_dim]
        
        # Spectrogram
        spec_att = self.spec_attention(spec_x)
        spec_x = spec_x * spec_att
        
        # Feature Fusion
        fused_x = torch.cat([eeg_x, spec_x], dim=1)
        
        # Fusion Layer
        fused_x = F.relu(self.fusion_bn1(self.fusion_fc1(fused_x)))
        fused_x = self.fusion_dropout1(fused_x)
        
        fused_x = F.relu(self.fusion_bn2(self.fusion_fc2(fused_x)))
        fused_x = self.fusion_dropout2(fused_x)
        
        fused_x = F.relu(self.fusion_bn3(self.fusion_fc3(fused_x)))
        fused_x = self.fusion_dropout3(fused_x)
        
        output = self.output_fc(fused_x)
        
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use equipment: {device}")

# Load the trained model
model = MultiModalEfficientNet(backbone='efficientnet_b2').to(device)
model.load_state_dict(torch.load('/kaggle/input/3th/pytorch/default/1/best_efficientnet_eeg_model.pth', map_location=device))
model.eval()
print("Model loading completed")

# Creating a test dataset
test_dataset = EEGTestSpectrogramDataset(
    csv_file="/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    eeg_dir="/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
)

# Creating a test data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=12,  # Keep it consistent with your training
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Test dataset size: {len(test_dataset)}")

# Generate predictions
predictions = []
eeg_ids = []

print("Start generating forecasts...")
with torch.no_grad():
    for batch_idx, (eeg_batch, spec_batch, id_batch) in enumerate(test_loader):
        eeg_batch = eeg_batch.to(device)
        spec_batch = spec_batch.to(device)
        
        # Model predictions
        logits = model(eeg_batch, spec_batch)
        probabilities = torch.softmax(logits, dim=1)
        
        # Save prediction results
        predictions.extend(probabilities.cpu().numpy())
        eeg_ids.extend(id_batch)
        
        # Print progress
        if batch_idx % 50 == 0:
            print(f"Processing progress: {batch_idx}/{len(test_loader)} batches")

print("Prediction completed！")

# Read the original CSV file of the test data to get the correct order
test_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")

# Create a prediction result DataFrame
predictions_array = np.array(predictions)
eeg_ids_array = np.array(eeg_ids)

# Create a DataFrame to match eeg_id and prediction results
pred_df = pd.DataFrame({
    'eeg_id': eeg_ids_array,
    'predictions': list(predictions_array)
})

# Category Name
class_names = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

# Expand the prediction results into columns
for i, class_name in enumerate(class_names):
    pred_df[class_name] = pred_df['predictions'].apply(lambda x: x[i])

# Delete temporary columns
pred_df = pred_df.drop('predictions', axis=1)

# Merge with original test data to ensure correct order
submission_df = test_df[['eeg_id']].merge(pred_df, on='eeg_id', how='left')

# Check if there are any missing values and fill them with the mean value if there are any
if submission_df[class_names].isnull().any().any():
    print("Find missing values and fill them with default values...")
    default_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  
    for i, class_name in enumerate(class_names):
        submission_df[class_name].fillna(default_probs[i], inplace=True)

# Rearrange the order of columns to match the submission format
submission_df = submission_df[['eeg_id'] + class_names]

# Save submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission saved as submission.csv")
print(f"Submit File Shape: {submission_df.shape}")
print("Preview of the first 5 lines:")
print(submission_df.head())

# Verify the correctness of the submitted file
print("\nVerify submission documents:")
print(f"Number of EEG IDs: {len(submission_df)}")
print(f"列名: {list(submission_df.columns)}")
print(f"Probability and inspection (first 5 rows):")
prob_sums = submission_df[class_names].sum(axis=1)
print(prob_sums.head())
print(f"The range of probability and: {prob_sums.min():.4f} - {prob_sums.max():.4f}")

print("\nSubmission completed！")
