import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataloader import DiseaseDataset
from utils.config import DataConfig, VaccineData
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

# Use all features from the original config
class FullFeatureConfig:
    STATIC_VAR_LIST = VaccineData.STATIC_VAR_LIST
    TIME_KNOWN_VAR_LIST = VaccineData.TIME_KNOWN_VAR_LIST
    TIME_UNKNOWN_VAR_LIST = VaccineData.TIME_UNKNOWN_VAR_LIST
    
    def get_variable(self, var_name):
        return getattr(self, var_name)

# File paths
static_file = os.path.join(os.path.dirname(__file__), "data", "static_data_COVID-19.csv")
known_file = os.path.join(os.path.dirname(__file__), "data", "time_dependent_known_data_COVID-19.csv")
unknown_file = os.path.join(os.path.dirname(__file__), "data", "time_dependent_unknown_data_COVID-19.csv")

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
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = None
else:
    val_size = max(2, len(dataset) // 5)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

class FullFeatureVSN(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.grns = nn.ModuleList([nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        ) for _ in range(num_features)])
        self.softmax = nn.Softmax(dim=2)
        self.weight_layer = nn.Linear(num_features * hidden_size, num_features)

    def forward(self, x):
        # x: [batch, time, num_features]
        batch, time, num_features = x.shape
        var_outputs = []
        for i in range(num_features):
            var = x[:, :, i:i+1]  # [batch, time, 1]
            var_out = self.grns[i](var)
            var_outputs.append(var_out)
        var_outputs = torch.stack(var_outputs, dim=2)  # [batch, time, num_features, hidden_size]
        flat = var_outputs.reshape(batch, time, -1)
        weights = self.softmax(self.weight_layer(flat)).unsqueeze(-1)  # [batch, time, num_features, 1]
        selected = (var_outputs * weights).sum(dim=2)  # [batch, time, hidden_size]
        return selected, weights

class FullFeatureSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch, time, d_model]
        batch, time, d_model = x.shape
        
        Q = self.query(x).view(batch, time, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch, time, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch, time, self.n_heads, self.d_k).transpose(1, 2)
        
        # [batch, n_heads, time, time]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)
        
        # [batch, n_heads, time, d_k]
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, d_model)
        output = self.output(attended)
        
        return output, attn_weights

class FullFeatureObjectiveLayer(nn.Module):
    def __init__(self, input_size, objectives, feature_objective_mapping, feature_names):
        super().__init__()
        self.objectives = objectives
        self.feature_objective_mapping = feature_objective_mapping
        self.feature_names = feature_names
        
        # Learnable feature weights for each objective
        self.feature_weights = nn.ParameterDict({
            obj: nn.Parameter(torch.ones(len(feature_objective_mapping[obj]))) for obj in objectives
        })
        
        # Per-objective heads with more capacity
        self.heads = nn.ModuleDict({
            obj: nn.Sequential(
                nn.Linear(len(feature_objective_mapping[obj]), 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ) for obj in objectives
        })
        
        # Equity/shared head
        self.equity_head = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: [batch, features] (should be the full feature vector)
        outputs = {}
        feature_importances = {}
        
        for obj in self.objectives:
            # Get indices for features relevant to this objective
            obj_features = self.feature_objective_mapping[obj]
            indices = []
            for f in obj_features:
                if f in self.feature_names:
                    indices.append(self.feature_names.index(f))
            
            if indices:  # Only process if we have relevant features
                x_obj = x[:, indices]  # [batch, num_obj_features]
                weights = torch.softmax(self.feature_weights[obj][:len(indices)], dim=0)
                feature_importances[obj] = weights.detach().cpu().numpy()
                x_weighted = x_obj * weights  # [batch, num_obj_features]
                outputs[obj] = self.heads[obj](x_weighted)
            else:
                # Fallback: use all features if no specific features found
                weights = torch.softmax(torch.ones(x.size(1)), dim=0)
                feature_importances[obj] = weights.detach().cpu().numpy()
                outputs[obj] = self.heads[obj](x)
        
        equity_score = self.equity_head(x)
        return outputs, feature_importances, equity_score

class FullFeatureTFT(nn.Module):
    def __init__(self, known_features, static_features, unknown_features, objectives, feature_objective_mapping, feature_names):
        super().__init__()
        self.known_features = known_features
        self.static_features = static_features
        self.unknown_features = unknown_features
        
        # Variable Selection Networks
        self.known_vsn = FullFeatureVSN(known_features, hidden_size=32)
        self.static_vsn = FullFeatureVSN(static_features, hidden_size=32)
        self.unknown_vsn = FullFeatureVSN(unknown_features, hidden_size=32)
        
        # Static context encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(96, 64, num_layers=2, batch_first=True, dropout=0.1)
        
        # Self-attention
        self.attention = FullFeatureSelfAttention(64, n_heads=4)
        
        # Final processing
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Objective layer - use all features (known + static + unknown)
        total_features = known_features + static_features + unknown_features
        self.objective_layer = FullFeatureObjectiveLayer(
            total_features, objectives, feature_objective_mapping, feature_names
        )

    def forward(self, x_known, x_static, x_unknown):
        # Variable selection
        known_selected, known_weights = self.known_vsn(x_known)
        static_selected, static_weights = self.static_vsn(x_static)
        unknown_selected, unknown_weights = self.unknown_vsn(x_unknown)
        
        # Static context
        static_context = self.static_encoder(static_selected[:, 0, :])  # Use first timestep
        static_context_expanded = static_context.unsqueeze(1).repeat(1, known_selected.size(1), 1)
        
        # Combine all features
        combined = torch.cat([known_selected, static_context_expanded, unknown_selected], dim=2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        
        # Self-attention
        attended, attn_weights = self.attention(lstm_out)
        
        # Final processing
        out = attended[:, -1, :]  # Use last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # For objective layer, concatenate all features from last timestep
        feature_vector = torch.cat([
            x_known[:, -1, :],  # Last timestep of known features
            x_static[:, 0, :],  # Static features (same for all timesteps)
            x_unknown[:, -1, :]  # Last timestep of unknown features
        ], dim=1)
        
        outputs, feature_importances, equity_score = self.objective_layer(feature_vector)
        
        return outputs, feature_importances, equity_score, {
            'known_weights': known_weights,
            'static_weights': static_weights,
            'unknown_weights': unknown_weights,
            'attention_weights': attn_weights
        }

# Get all feature names in order
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

# Initialize model
model = FullFeatureTFT(
    known_features=len(config.TIME_KNOWN_VAR_LIST),
    static_features=len(config.STATIC_VAR_LIST),
    unknown_features=len(config.TIME_UNKNOWN_VAR_LIST),
    objectives=objectives,
    feature_objective_mapping=feature_objective_mapping,
    feature_names=all_feature_names
)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# For visualization: track per-objective output means and feature importances
objective_output_stats = {obj: [] for obj in objectives}
feature_importance_stats = {obj: [] for obj in objectives}
variable_selection_stats = {
    'known': [],
    'static': [],
    'unknown': []
}

print(f"\n=== TRAINING ===")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(train_loader)}")
if val_loader:
    print(f"Validation samples: {len(val_loader)}")

for epoch in range(30):  # Increased epochs for full feature model
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        x_known = batch['inputs']['known']
        x_static = batch['inputs']['static']
        x_unknown = batch['inputs']['unknown']
        y = batch['targets'][:, :len(objectives)]  # Use as many columns as objectives
        
        optimizer.zero_grad()
        
        outputs_dict, feature_importances, equity_score, weights = model(x_known, x_static, x_unknown)
        preds = torch.cat([outputs_dict[obj] for obj in objectives], dim=1)
        
        loss = criterion(preds, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Collect statistics for the first batch only
        if batch_idx == 0:
            for obj in objectives:
                objective_output_stats[obj].append(preds[:, objectives.index(obj)].mean().item())
                feature_importance_stats[obj].append(feature_importances[obj])
            
            # Track variable selection weights
            variable_selection_stats['known'].append(weights['known_weights'].mean(dim=(0,1,3)).detach().cpu().numpy())
            variable_selection_stats['static'].append(weights['static_weights'].mean(dim=(0,1,3)).detach().cpu().numpy())
            variable_selection_stats['unknown'].append(weights['unknown_weights'].mean(dim=(0,1,3)).detach().cpu().numpy())
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    if val_loader is not None:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_known = batch['inputs']['known']
                x_static = batch['inputs']['static']
                x_unknown = batch['inputs']['unknown']
                y = batch['targets'][:, :len(objectives)]
                
                outputs_dict, _, _, _ = model(x_known, x_static, x_unknown)
                preds = torch.cat([outputs_dict[obj] for obj in objectives], dim=1)
                loss = criterion(preds, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    else:
        print(f"Epoch {epoch+1:2d}, Train Loss: {avg_train_loss:.4f}")

print(f"\n=== RESULTS ===")

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

print(f"\n=== SUMMARY ===")
print(f"Training completed with {len(objectives)} objectives")
print(f"Results saved to:")
print(f"  - full_feature_results.csv")
print(f"  - variable_selection_weights.csv")
print(f"  - objective_outputs_full_features.png")
print(f"  - feature_importances_full_features.png")
print(f"  - variable_selection_weights_full_features.png")
print(f"\nFor production use, consider:")
print(f"  - Collecting {recommended_samples:.0f} samples minimum")
print(f"  - Using {comfortable_samples:.0f} samples for robust performance")
print(f"  - Implementing data augmentation techniques")
print(f"  - Cross-validation with multiple folds") 