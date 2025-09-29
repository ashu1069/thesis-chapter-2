import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataloader import DiseaseDataset
from utils.config import DataConfig, VaccineData
from models.tft import TemporalFusionTransformer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Suppress warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Use all features from the original config
class FullFeatureConfig:
    STATIC_VAR_LIST = VaccineData.STATIC_VAR_LIST
    TIME_KNOWN_VAR_LIST = VaccineData.TIME_KNOWN_VAR_LIST
    TIME_UNKNOWN_VAR_LIST = VaccineData.TIME_UNKNOWN_VAR_LIST
    
    def get_variable(self, var_name):
        return getattr(self, var_name)

# File paths (override via env vars if provided)
default_data_dir = os.path.join(os.path.dirname(__file__), "data")
static_file = os.environ.get("STATIC_CSV", os.path.join(default_data_dir, "static_data_synthetic.csv"))
known_file = os.environ.get("KNOWN_CSV", os.path.join(default_data_dir, "time_dependent_known_data_synthetic.csv"))
unknown_file = os.environ.get("UNKNOWN_CSV", os.path.join(default_data_dir, "time_dependent_unknown_data_synthetic.csv"))

config = FullFeatureConfig()
dataset = DiseaseDataset(static_file, known_file, unknown_file, config)

print(f"=== DATA ANALYSIS ===")
print(f"Current dataset size: {len(dataset)} samples")
print(f"Static features: {len(config.STATIC_VAR_LIST)}")
print(f"Time-known features: {len(config.TIME_KNOWN_VAR_LIST)}")
print(f"Time-unknown features: {len(config.TIME_UNKNOWN_VAR_LIST)}")
print(f"Total features: {len(config.STATIC_VAR_LIST) + len(config.TIME_KNOWN_VAR_LIST) + len(config.TIME_UNKNOWN_VAR_LIST)}")

# Calculate ideal data size recommendations
total_features = len(config.STATIC_VAR_LIST) + len(config.TIME_KNOWN_VAR_LIST) + len(config.TIME_UNKNOWN_VAR_LIST)
num_objectives = len(VaccineData.FEATURE_OBJECTIVE_MAPPING)

print(f"\n=== IDEAL DATA SIZE RECOMMENDATIONS ===")
print(f"Number of objectives: {num_objectives}")

# Rule of thumb: 10-20 samples per parameter for robust training
# For a TFT model with variable selection and attention mechanisms
estimated_params_per_objective = 50  # Rough estimate for the objective-specific networks
total_estimated_params = estimated_params_per_objective * num_objectives + total_features * 10  # Additional params for feature processing

min_samples_rule_of_thumb = total_estimated_params * 10
recommended_samples = total_estimated_params * 20
comfortable_samples = total_estimated_params * 50

print(f"Estimated parameters per objective: ~{estimated_params_per_objective}")
print(f"Total estimated parameters: ~{total_estimated_params}")
print(f"Minimum samples (10x params): {min_samples_rule_of_thumb:.0f}")
print(f"Recommended samples (20x params): {recommended_samples:.0f}")
print(f"Comfortable samples (50x params): {comfortable_samples:.0f}")

# For time series, also consider sequence length
sequence_length = 5  # Current window size
print(f"\nTime series considerations:")
print(f"Sequence length: {sequence_length}")
print(f"Effective samples with sequences: {len(dataset) * sequence_length}")
print(f"Recommended time steps: {recommended_samples * sequence_length:.0f}")

# Data augmentation recommendations
print(f"\n=== DATA AUGMENTATION RECOMMENDATIONS ===")
if len(dataset) < min_samples_rule_of_thumb:
    augmentation_factor = math.ceil(min_samples_rule_of_thumb / len(dataset))
    print(f"Current data is insufficient. Recommended augmentation factor: {augmentation_factor}x")
    print(f"Methods: synthetic data generation, bootstrapping, domain-specific augmentation")
else:
    print(f"Current data size is adequate for initial training")

# Duplicate data if too small for a split
min_samples = 20  # Increased minimum for full feature model
if len(dataset) < min_samples:
    print(f"\n[INFO] Duplicating data to reach at least {min_samples} samples for splitting.")
    items = [dataset[i] for i in range(len(dataset))]
    repeats = (min_samples + len(items) - 1) // len(items)
    all_items = items * repeats
    all_items = all_items[:min_samples]
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]
    dataset = ListDataset(all_items)

# Validation split
if len(dataset) < 5:
    print(f"[WARNING] Dataset too small for train/val split (len={len(dataset)}). Using all data for training and skipping validation.")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = None
else:
    val_size = max(2, len(dataset) // 5)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

# Custom TFT classes removed - using utils.modules.TemporalFusionTransformer instead

# Get all feature names in order
all_feature_names = (
    config.TIME_KNOWN_VAR_LIST + 
    config.Static_VAR_LIST if hasattr(config, 'Static_VAR_LIST') else []
)
all_feature_names = (
    config.TIME_KNOWN_VAR_LIST + 
    config.STATIC_VAR_LIST + 
    config.TIME_UNKNOWN_VAR_LIST
)

# Filter objectives to only include those with available features
objectives = []
feature_objective_mapping = {}
for obj in VaccineData.FEATURE_OBJECTIVE_MAPPING:
    filtered_feats = [f for f in VaccineData.FEATURE_OBJECTIVE_MAPPING[obj] if f in all_feature_names]
    if filtered_feats:
        objectives.append(obj)
        feature_objective_mapping[obj] = filtered_feats

print(f"\n=== MODEL CONFIGURATION ===")
print(f"Objectives used for training: {objectives}")
print(f"Feature counts:")
print(f"  Known features: {len(config.TIME_KNOWN_VAR_LIST)}")
print(f"  Static features: {len(config.STATIC_VAR_LIST)}")
print(f"  Unknown features: {len(config.TIME_UNKNOWN_VAR_LIST)}")
print(f"  Total features: {len(all_feature_names)}")

# Initialize model with proper TFT configuration
def _sizes_dict(names):
    return {name: 1 for name in names}

combined_time_names = config.TIME_KNOWN_VAR_LIST + config.TIME_UNKNOWN_VAR_LIST

tft_config = {
    'hidden_size': 64,
    'nhead': 4,
    'dropout': 0.1,
    'ff_hidden_size': 128,
    'lstm_hidden_size': 64,
    'context_size': 64,
    'num_lstm_layers': 2,
    'num_attention_layers': 2,
    'num_layers': 2,
    'historical_length': 5,
    'forecast_length': 5,
    'static_input_sizes': _sizes_dict(config.STATIC_VAR_LIST),
    'historical_input_sizes': _sizes_dict(combined_time_names),
    'future_input_sizes': _sizes_dict(combined_time_names),
    'output_size': len(objectives)
}

model = TemporalFusionTransformer(tft_config)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()  # Mean Absolute Error

# For visualization: track per-objective output means and feature importances
objective_output_stats = {obj: [] for obj in objectives}
feature_importance_stats = {obj: [] for obj in objectives}
variable_selection_stats = {
    'known': [],
    'static': [],
    'unknown': []
}

print(f"\n{'='*60}")
print(f"🚀 STARTING TFT TRAINING")
print(f"{'='*60}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"📈 Training batches: {len(train_loader)} ({len(train_loader) * 64:,} samples)")
if val_loader:
    print(f"📈 Validation batches: {len(val_loader)} ({len(val_loader) * 64:,} samples)")
print(f"🎯 Objectives: {len(objectives)}")
print(f"⚙️  Batch size: 64 | Learning rate: 1e-4 | Epochs: 50")
print(f"{'='*60}")

# Prepare checkpoint directory
ckpt_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(ckpt_dir, exist_ok=True)
best_val_loss = float('inf')
best_ckpt_path = None

# Early stopping and learning rate scheduling
patience = 10  # epochs to wait before reducing LR
lr_patience = 5  # epochs to wait before early stopping
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
early_stopping_counter = 0

for epoch in range(50):  # More epochs for robust training
    model.train()
    total_loss = 0
    total_mae = 0
    
    iterator = train_loader if tqdm is None else tqdm(train_loader, desc=f"Epoch {epoch+1}/50 [train]", leave=False, ncols=80)
    for batch_idx, batch in enumerate(iterator):
        # Prepare input for TFT
        # Split multivariate tensors into per-variable 1D tensors keyed by names
        static_dict = {name: batch['inputs']['static'][:, 0, i:i+1] for i, name in enumerate(config.STATIC_VAR_LIST)}
        known_dict = {name: batch['inputs']['known'][:, :, i:i+1] for i, name in enumerate(config.TIME_KNOWN_VAR_LIST)}
        unknown_dict = {name: batch['inputs']['unknown'][:, :, i:i+1] for i, name in enumerate(config.TIME_UNKNOWN_VAR_LIST)}
        all_time_vars = {**known_dict, **unknown_dict}

        x = {'static': static_dict, 'historical': all_time_vars, 'future': all_time_vars}
        y = batch['targets'][:, :len(objectives)]  # Use as many columns as objectives
        
        optimizer.zero_grad()
        
        outputs = model(x)
        preds = outputs['objective_tensor']
        
        loss = criterion(preds, y)
        mae = mae_criterion(preds, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae.item()
        
        # Collect statistics for the first batch only
        if batch_idx == 0:
            for i, obj in enumerate(objectives):
                objective_output_stats[obj].append(preds[:, i].mean().item())
                # Simplified feature importance tracking
                feature_importance_stats[obj].append(np.ones(len(config.STATIC_VAR_LIST)) / len(config.STATIC_VAR_LIST))
            
            # Simplified variable selection weights tracking
            variable_selection_stats['known'].append(np.ones(len(config.TIME_KNOWN_VAR_LIST)) / len(config.TIME_KNOWN_VAR_LIST))
            variable_selection_stats['static'].append(np.ones(len(config.STATIC_VAR_LIST)) / len(config.STATIC_VAR_LIST))
            variable_selection_stats['unknown'].append(np.ones(len(config.TIME_UNKNOWN_VAR_LIST)) / len(config.TIME_UNKNOWN_VAR_LIST))
        
        if tqdm is not None:
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    avg_train_mae = total_mae / len(train_loader)
    
    # Validation
    if val_loader is not None:
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            viterator = val_loader if tqdm is None else tqdm(val_loader, desc=f"Epoch {epoch+1}/50 [val]", leave=False, ncols=80)
            for batch in viterator:
                # Prepare input for TFT
                static_dict = {name: batch['inputs']['static'][:, 0, i:i+1] for i, name in enumerate(config.STATIC_VAR_LIST)}
                known_dict = {name: batch['inputs']['known'][:, :, i:i+1] for i, name in enumerate(config.TIME_KNOWN_VAR_LIST)}
                unknown_dict = {name: batch['inputs']['unknown'][:, :, i:i+1] for i, name in enumerate(config.TIME_UNKNOWN_VAR_LIST)}
                all_time_vars = {**known_dict, **unknown_dict}
                x = {'static': static_dict, 'historical': all_time_vars, 'future': all_time_vars}
                y = batch['targets'][:, :len(objectives)]
                
                outputs = model(x)
                preds = outputs['objective_tensor']
                loss = criterion(preds, y)
                mae = mae_criterion(preds, y)
                val_loss += loss.item()
                val_mae += mae.item()
                if tqdm is not None:
                    viterator.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        print(f"📊 Epoch {epoch+1:2d}/50 | Train: {avg_train_loss:.4f} (MAE: {avg_train_mae:.4f}) | Val: {avg_val_loss:.4f} (MAE: {avg_val_mae:.4f})")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best checkpoint (by val loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # Reset counter
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'objectives': objectives,
                'feature_objective_mapping': feature_objective_mapping,
                'feature_names': all_feature_names,
            }
            best_ckpt_path = os.path.join(ckpt_dir, 'tft_full_features_best.pt')
            torch.save(ckpt, best_ckpt_path)
            print(f"💾 [BEST] Model saved to {best_ckpt_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= lr_patience:
                print(f"🛑 Early stopping triggered after {lr_patience} epochs without improvement")
                break
    else:
        print(f"📊 Epoch {epoch+1:2d}/50 | Train: {avg_train_loss:.4f} (MAE: {avg_train_mae:.4f})")

print(f"\n{'='*60}")
print(f"🎉 TRAINING COMPLETED")
print(f"{'='*60}")

# Plot per-objective output means
plt.figure(figsize=(12, 6))
for obj in objectives:
    plt.plot(objective_output_stats[obj], label=f"{obj} mean output", linewidth=2)
plt.legend()
plt.title('Per-Objective Output Means Over Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Output (normalized)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('objective_outputs_full_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot feature importances for each objective (last epoch)
plt.figure(figsize=(15, 8))
for i, obj in enumerate(objectives):
    plt.subplot(2, 2, i+1)
    importances = feature_importance_stats[obj][-1]
    features = feature_objective_mapping[obj]
    plt.barh(features, importances, alpha=0.7)
    plt.title(f'{obj}\nFeature Importances')
    plt.xlabel('Importance (softmax weight)')
plt.tight_layout()
plt.savefig('feature_importances_full_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot variable selection weights
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (var_type, feature_list) in enumerate([
    ('known', config.TIME_KNOWN_VAR_LIST),
    ('static', config.STATIC_VAR_LIST),
    ('unknown', config.TIME_UNKNOWN_VAR_LIST)
]):
    weights_array = np.array(variable_selection_stats[var_type])
    for j, feature in enumerate(feature_list):
        axes[i].plot(weights_array[:, j], label=feature, alpha=0.7)
    axes[i].set_title(f'{var_type.capitalize()} Variable Selection Weights')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Weight')
    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('variable_selection_weights_full_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final feature importances
print("\nFinal Feature Importances per objective:")
for obj in objectives:
    print(f"\n{obj}:")
    importances = feature_importance_stats[obj][-1]
    features = feature_objective_mapping[obj]
    for fname, weight in zip(features, importances):
        print(f"  {fname}: {weight:.4f}")

# Save results to CSV
with open('full_feature_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Objective', 'Feature', 'Importance'])
    for obj in objectives:
        importances = feature_importance_stats[obj][-1]
        features = feature_objective_mapping[obj]
        for fname, weight in zip(features, importances):
            writer.writerow([obj, fname, weight])

# Save variable selection weights
with open('variable_selection_weights.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Variable_Type', 'Feature', 'Epoch', 'Weight'])
    
    for var_type, feature_list in [
        ('known', config.TIME_KNOWN_VAR_LIST),
        ('static', config.STATIC_VAR_LIST),
        ('unknown', config.TIME_UNKNOWN_VAR_LIST)
    ]:
        weights_array = np.array(variable_selection_stats[var_type])
        for epoch in range(weights_array.shape[0]):
            for feature, weight in zip(feature_list, weights_array[epoch]):
                writer.writerow([var_type, feature, epoch, weight])

print(f"\n📈 TRAINING SUMMARY")
print(f"✅ Completed training with {len(objectives)} objectives")
print(f"📁 Results saved to:")
print(f"   📄 full_feature_results.csv")
print(f"   📄 variable_selection_weights.csv")
print(f"   📊 objective_outputs_full_features.png")
print(f"   📊 feature_importances_full_features.png")
print(f"   📊 variable_selection_weights_full_features.png")
print(f"   💾 models/tft_full_features_best.pt")
print(f"\n💡 For production use, consider:")
print(f"   📊 Collecting {recommended_samples:.0f} samples minimum")
print(f"   📊 Using {comfortable_samples:.0f} samples for robust performance")
print(f"   🔄 Implementing data augmentation techniques")
print(f"   🔄 Cross-validation with multiple folds")
print(f"{'='*60}") 