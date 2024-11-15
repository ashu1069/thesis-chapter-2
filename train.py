import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vaccine_prioritization.models.tft import TemporalFusionTransformer
from vaccine_prioritization.utils.loss import custom_loss

# Configuration dictionary
config = {
    'static_variables': [
        'Endemic_Potential_R0', 
        'Endemic_Potential_Duration', 
        'Demography_Urban_Rural_Split', 
        'Demography_Population_Density', 
        'Environmental_Index', 
        'Socio_economic_Gini_Index', 
        'Socio_economic_Poverty_Rates',
        'Communication_Affordability',
        'Socio_economic_GDP_per_capita',
        'Socio_economic_Employment_Rates',
        'Socio_economic_Education_Levels'
    ],
    'historical_variables': [
        'Healthcare_Index_Tier_X_hospitals',
        'Healthcare_Index_Workforce_capacity',
        'Healthcare_Index_Bed_availability_per_capita',
        'Healthcare_Index_Expenditure_per_capita',
        'Immunization_Coverage',
        'Economic_Index_Budget_allocation_per_capita',
        'Economic_Index_Fraction_of_total_budget',
        'Political_Stability_Index'
    ],
    'future_variables': [
        'Frequency_of_outbreaks',
        'Magnitude_of_outbreaks_Deaths',
        'Magnitude_of_outbreaks_Infected',
        'Magnitude_of_outbreaks_Severity_Index',
        'Security_and_Conflict_Index'
    ],
    'static_input_sizes': {
        'Endemic_Potential_R0': 1,
        'Endemic_Potential_Duration': 1,
        'Demography_Urban_Rural_Split': 1,
        'Demography_Population_Density': 1,
        'Environmental_Index': 1,
        'Socio_economic_Gini_Index': 1,
        'Socio_economic_Poverty_Rates': 1,
        'Communication_Affordability': 1,
        'Socio_economic_GDP_per_capita': 1,
        'Socio_economic_Employment_Rates': 1,
        'Socio_economic_Education_Levels': 1
    },
    'historical_input_sizes': {
        'Healthcare_Index_Tier_X_hospitals': 1,
        'Healthcare_Index_Workforce_capacity': 1,
        'Healthcare_Index_Bed_availability_per_capita': 1,
        'Healthcare_Index_Expenditure_per_capita': 1,
        'Immunization_Coverage': 1,
        'Economic_Index_Budget_allocation_per_capita': 1,
        'Economic_Index_Fraction_of_total_budget': 1,
        'Political_Stability_Index': 1
    },
    'future_input_sizes': {
        'Frequency_of_outbreaks': 1,
        'Magnitude_of_outbreaks_Deaths': 1,
        'Magnitude_of_outbreaks_Infected': 1,
        'Magnitude_of_outbreaks_Severity_Index': 1,
        'Security_and_Conflict_Index': 1
    },
    'hidden_size': 16,
    'lstm_hidden_size': 16,
    'ff_hidden_size': 16,
    'dropout': 0.1,
    'nhead': 4,
    'num_lstm_layers': 1,
    'num_attention_layers': 1,
    'num_layers': 1,
    'output_size': 4,
    'objective_weights': [0.30, 0.30, 0.25, 0.15],
    'feature_objective_mapping': {
        'Maximize Health Impact': ['Endemic_Potential_R0', 'Endemic_Potential_Duration', 'Healthcare_Index_Tier_X_hospitals', 'Healthcare_Index_Workforce_capacity', 'Healthcare_Index_Bed_availability_per_capita', 'Immunization_Coverage', 'Frequency_of_outbreaks', 'Magnitude_of_outbreaks_Deaths', 'Magnitude_of_outbreaks_Infected', 'Magnitude_of_outbreaks_Severity_Index'],
        'Maximize Value for Money': ['Economic_Index_Budget_allocation_per_capita', 'Economic_Index_Fraction_of_total_budget', 'Healthcare_Index_Expenditure_per_capita', 'Socio_economic_GDP_per_capita', 'Socio_economic_Employment_Rates', 'Socio_economic_Education_Levels'],
        'Reinforce Financial Sustainability': ['Economic_Index_Budget_allocation_per_capita', 'Economic_Index_Fraction_of_total_budget', 'Healthcare_Index_Expenditure_per_capita', 'Socio_economic_GDP_per_capita', 'Socio_economic_Employment_Rates', 'Socio_economic_Poverty_Rates', 'Political_Stability_Index', 'Communication_Affordability'],
        'Support Countries with the Greatest Needs': ['Demography_Urban_Rural_Split', 'Demography_Population_Density', 'Environmental_Index', 'Socio_economic_Gini_Index', 'Socio_economic_Poverty_Rates', 'Healthcare_Index_Bed_availability_per_capita', 'Political_Stability_Index', 'Security_and_Conflict_Index']
    },
    'context_size': 16,
    'variable_embedding_size': 16,
    'embedding_size': 16,
    'grn_hidden': 16,
    'static_embedding_size': 16,
    'temporal_embedding_size': 16,
    'time_embedding_size': 16,
    'categorical_embedding_size': 16,
    'gating_size': 16,
    'historical_length': 5,
    'forecast_length': 5,
}

# Add these dimension calculations to config
config['total_embedding_size'] = config['hidden_size']
config['input_embedding_size'] = config['hidden_size']

# Add sequence information to config
config['sequence_lengths'] = {
    'historical': 5,
    'future': 5
}

# Calculate total input sizes after config creation
config['input_size_total'] = {
    'static': sum(config['static_input_sizes'].values()),
    'historical': sum(config['historical_input_sizes'].values()),
    'future': sum(config['future_input_sizes'].values())
}

# Example data (updated to match config)
x_static = {
    k: torch.randn(100, config['static_input_sizes'][k]) 
    for k in config['static_variables']
}

x_historical = {
    k: torch.randn(100, 5, config['historical_input_sizes'][k])  # 5 time steps
    for k in config['historical_variables']
}

x_future = {
    k: torch.randn(100, 5, config['future_input_sizes'][k])  # 5 time steps
    for k in config['future_variables']
}

y = torch.randn(100, 4)  # 100 samples, 4 objectives

# Example validation data (replace with your actual data)
x_static_val = {k: v[:20] for k, v in x_static.items()}
x_historical_val = {k: v[:20] for k, v in x_historical.items()}
x_future_val = {k: v[:20] for k, v in x_future.items()}
y_val = y[:20]

# Create DataLoader
dataset = TensorDataset(
    torch.cat([x_static[k] for k in config['static_variables']], dim=1),
    torch.cat([x_historical[k] for k in config['historical_variables']], dim=2),
    torch.cat([x_future[k] for k in config['future_variables']], dim=2),
    y
)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create validation DataLoader
val_dataset = TensorDataset(
    torch.cat([x_static_val[k] for k in config['static_variables']], dim=1),
    torch.cat([x_historical_val[k] for k in config['historical_variables']], dim=2),
    torch.cat([x_future_val[k] for k in config['future_variables']], dim=2),
    y_val
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, optimizer
model = TemporalFusionTransformer(config)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x_static_batch, x_historical_batch, x_future_batch, y = batch
        
        x = {
            'static': {},
            'historical': {},
            'future': {}
        }
        
        # Process static variables
        start_idx = 0
        for var in config['static_variables']:
            size = config['static_input_sizes'][var]
            x['static'][var] = x_static_batch[:, start_idx:start_idx + size]
            start_idx += size
        
        # Process historical variables
        start_idx = 0
        for var in config['historical_variables']:
            size = config['historical_input_sizes'][var]
            x['historical'][var] = x_historical_batch[:, :, start_idx:start_idx + size]
            start_idx += size
        
        # Process future variables
        start_idx = 0
        for var in config['future_variables']:
            size = config['future_input_sizes'][var]
            x['future'][var] = x_future_batch[:, :, start_idx:start_idx + size]
            start_idx += size

        optimizer.zero_grad()
        output = model(x)
        loss = custom_loss(
            model_outputs=output,
            targets=y,
            objective_weights=config['objective_weights']
        )
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            x_static_batch, x_historical_batch, x_future_batch, y = batch
            
            x = {
                'static': {},
                'historical': {},
                'future': {}
            }
            
            # Process static variables
            start_idx = 0
            for var in config['static_variables']:
                size = config['static_input_sizes'][var]
                x['static'][var] = x_static_batch[:, start_idx:start_idx + size]
                start_idx += size
            
            # Process historical variables
            start_idx = 0
            for var in config['historical_variables']:
                size = config['historical_input_sizes'][var]
                x['historical'][var] = x_historical_batch[:, :, start_idx:start_idx + size]
                start_idx += size
            
            # Process future variables
            start_idx = 0
            for var in config['future_variables']:
                size = config['future_input_sizes'][var]
                x['future'][var] = x_future_batch[:, :, start_idx:start_idx + size]
                start_idx += size

            output = model(x)
            val_loss += custom_loss(
                model_outputs=output,
                targets=y,
                objective_weights=config['objective_weights']
            ).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')
    
    print(f'Epoch {epoch}, Training Loss: {loss.item()}')