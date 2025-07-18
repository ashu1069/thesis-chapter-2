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

# Expanded config: 3 known features, 2 targets, 2 static features
class MinimalConfig:
    STATIC_VAR_LIST = [
        'Endemic_Potential_R0',
        'Endemic_Potential_Duration',
    ]
    TIME_KNOWN_VAR_LIST = [
        'Healthcare_Index_Tier_X_hospitals',
        'Healthcare_Index_Workforce_capacity',
        'Healthcare_Index_Bed_availability_per_capita',
    ]
    TIME_UNKNOWN_VAR_LIST = [
        'Frequency_of_outbreaks',
        'Magnitude_of_outbreaks_Deaths',
    ]
    def get_variable(self, var_name):
        return getattr(self, var_name)

# File paths
static_file = os.path.join(os.path.dirname(__file__), "data", "static_data_COVID-19.csv")
known_file = os.path.join(os.path.dirname(__file__), "data", "time_dependent_known_data_COVID-19.csv")
unknown_file = os.path.join(os.path.dirname(__file__), "data", "time_dependent_unknown_data_COVID-19.csv")

config = MinimalConfig()
dataset = DiseaseDataset(static_file, known_file, unknown_file, config)

# Duplicate data if too small for a split
min_samples = 10
if len(dataset) < min_samples:
    print(f"[INFO] Duplicating data to reach at least {min_samples} samples for splitting.")
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
if len(dataset) < 3:
    print(f"[WARNING] Dataset too small for train/val split (len={len(dataset)}). Using all data for training and skipping validation.")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = None
else:
    val_size = max(1, len(dataset) // 5)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model: LSTM on known+static data, predict 2 targets
class SimpleVSN(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.grns = nn.ModuleList([nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
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
        return selected

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        # x: [batch, time, d_model]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # [batch, time, time]
        attn_weights = self.softmax(attn_scores)
        attended = torch.bmm(attn_weights, V)  # [batch, time, d_model]
        return attended, attn_weights

class ObjectiveLayer(nn.Module):
    def __init__(self, input_size, objectives, feature_objective_mapping, feature_names):
        super().__init__()
        self.objectives = objectives
        self.feature_objective_mapping = feature_objective_mapping
        self.feature_names = feature_names  # list of all feature names (order matches input)
        # Learnable feature weights for each objective
        self.feature_weights = nn.ParameterDict({
            obj: nn.Parameter(torch.ones(len(feature_objective_mapping[obj]))) for obj in objectives
        })
        # Per-objective heads
        self.heads = nn.ModuleDict({
            obj: nn.Sequential(
                nn.Linear(len(feature_objective_mapping[obj]), 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ) for obj in objectives
        })
        # Equity/shared head
        self.equity_head = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # x: [batch, features] (should be the full feature vector)
        outputs = {}
        feature_importances = {}
        for obj in self.objectives:
            indices = [self.feature_names.index(f) for f in self.feature_objective_mapping[obj]]
            x_obj = x[:, indices]  # [batch, num_obj_features]
            weights = torch.softmax(self.feature_weights[obj], dim=0)
            feature_importances[obj] = weights.detach().cpu().numpy()
            x_weighted = x_obj * weights  # [batch, num_obj_features]
            outputs[obj] = self.heads[obj](x_weighted)
        equity_score = self.equity_head(x)
        return outputs, feature_importances, equity_score

# Update model to use ObjectiveLayer with feature weighting and equity
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, objectives, feature_objective_mapping, feature_names):
        super().__init__()
        self.vsn = SimpleVSN(num_features=input_size, hidden_size=8)
        self.static_encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )
        self.lstm = nn.LSTM(16, 16, num_layers=2, batch_first=True)
        self.attn = SimpleSelfAttention(16)
        self.fc1 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        # The full feature vector will be [3 known + 2 static]
        self.objective_layer = ObjectiveLayer(5, objectives, feature_objective_mapping, feature_names)

    def forward(self, x, static):
        x_vsn = self.vsn(x)
        static_context = self.static_encoder(static)
        static_context_expanded = static_context.unsqueeze(1).repeat(1, x_vsn.size(1), 1)
        x_enriched = torch.cat([x_vsn, static_context_expanded], dim=2)
        out, _ = self.lstm(x_enriched)
        out_attn, attn_weights = self.attn(out)
        out = out_attn[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        # For feature weighting, pass the original input features (not the hidden)
        # Here, we use the last time step of known+static as the feature vector
        # (or you can use the mean over time, or concatenate static+known)
        # We'll use the last time step of known and static
        # x: [batch, time, 5], static: [batch, 2]
        x_last = x[:, -1, :]  # [batch, 5]
        outputs, feature_importances, equity_score = self.objective_layer(x_last)
        return outputs, feature_importances, equity_score, attn_weights

# Use the config's feature-objective mapping and feature names
feature_names = [
    'Healthcare_Index_Tier_X_hospitals',
    'Healthcare_Index_Workforce_capacity',
    'Healthcare_Index_Bed_availability_per_capita',
    'Endemic_Potential_R0',
    'Endemic_Potential_Duration',
]
objectives = []
feature_objective_mapping = {}
for obj in VaccineData.FEATURE_OBJECTIVE_MAPPING:
    filtered_feats = [f for f in VaccineData.FEATURE_OBJECTIVE_MAPPING[obj] if f in feature_names]
    if filtered_feats:
        objectives.append(obj)
        feature_objective_mapping[obj] = filtered_feats
print("Objectives used for training:", objectives)
print("Final feature_objective_mapping used for training:")
for obj, feats in feature_objective_mapping.items():
    print(f"{obj}: {feats}")
model = SimpleLSTM(input_size=5, objectives=objectives, feature_objective_mapping=feature_objective_mapping, feature_names=feature_names)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# For visualization: track per-objective output means and feature importances
objective_output_stats = {obj: [] for obj in objectives}
feature_importance_stats = {obj: [] for obj in objectives}

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x_known = batch['inputs']['known'][:, :, :3]
        static = batch['inputs']['static'][:, 0, :2]
        static_expanded = static.unsqueeze(1).repeat(1, x_known.size(1), 1)
        x = torch.cat([x_known, static_expanded], dim=2)
        y = batch['targets'][:, :len(objectives)]  # Use as many columns as objectives
        optimizer.zero_grad()

        outputs_dict, feature_importances, equity_score, _ = model(x, static)
        preds = torch.cat([outputs_dict[obj] for obj in objectives], dim=1)  # [batch, num_objectives]
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Collect per-objective output means and feature importances for the first batch only
        if batch_idx == 0:
            for obj in objectives:
                objective_output_stats[obj].append(preds[:, objectives.index(obj)].mean().item())
                feature_importance_stats[obj].append(feature_importances[obj])

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    if val_loader is not None:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_known = batch['inputs']['known'][:, :, :3]
                static = batch['inputs']['static'][:, 0, :2]
                static_expanded = static.unsqueeze(1).repeat(1, x_known.size(1), 1)
                x = torch.cat([x_known, static_expanded], dim=2)
                y = batch['targets'][:, :len(objectives)]
                outputs_dict, _, _, _ = model(x, static)
                preds = torch.cat([outputs_dict[obj] for obj in objectives], dim=1)
                loss = criterion(preds, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

# Plot per-objective output means after training
plt.figure(figsize=(10, 6))
for obj in objectives:
    plt.plot(objective_output_stats[obj], label=f"{obj} mean output")
plt.legend()
plt.title('Per-Objective Output Means Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Output (normalized)')
plt.show()

# Plot feature importances for each objective (last epoch)
plt.figure(figsize=(10, 6))
for obj in objectives:
    importances = feature_importance_stats[obj][-1]
    plt.bar([f for f in feature_objective_mapping[obj]], importances, label=obj, alpha=0.7)
plt.legend()
plt.title('Learned Feature Importances (Last Epoch)')
plt.ylabel('Importance (softmax weight)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print final feature importances for each objective
print("\nFinal Feature Importances (softmax weights) per objective:")
for obj in objectives:
    print(f"{obj}:")
    for fname, weight in zip(feature_objective_mapping[obj], feature_importance_stats[obj][-1]):
        print(f"  {fname}: {weight:.4f}")

# Save final feature importances to CSV
with open('feature_importances.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Objective', 'Feature', 'Importance'])
    for obj in objectives:
        for fname, weight in zip(feature_objective_mapping[obj], feature_importance_stats[obj][-1]):
            writer.writerow([obj, fname, weight])

# Plot feature importances over epochs for each objective
for obj in objectives:
    plt.figure(figsize=(8, 4))
    arr = np.stack(feature_importance_stats[obj])  # shape: [epochs, num_features]
    for i, fname in enumerate(feature_objective_mapping[obj]):
        plt.plot(arr[:, i], label=fname)
    plt.title(f'Feature Importances for {obj} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Importance (softmax weight)')
    plt.legend()
    plt.tight_layout()
    plt.show()