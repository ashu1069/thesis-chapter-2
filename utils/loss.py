import torch
import torch.nn.functional as F

def custom_loss(model_outputs, targets, objective_weights):
    """
    Custom loss function aligned with the TFT architecture and objective structure.
    
    Args:
        model_outputs (dict): Contains:
            - objective_outputs (dict): 
                - individual_scores (dict): Scores for each objective
                - equity_score (tensor): Score for equitable distribution
                - weighted_scores (dict): Weighted objective scores
                - final_score (tensor): Combined final score
            - weights (dict):
                - static (tensor): Feature weights for static variables
                - historical (tensor): Feature weights for historical variables
                - future (tensor): Feature weights for future variables
        targets (tensor): Target values for each objective [batch_size, num_objectives]
        objective_weights (list): Weights for each objective from config
        
    Returns:
        torch.Tensor: Combined loss value
    """
    # 1. Objective-specific prediction loss
    prediction_loss = 0.0
    for i, (objective, score) in enumerate(model_outputs['objective_outputs']['individual_scores'].items()):
        if objective != 'Promote Equitable Distribution':
            prediction_loss += objective_weights[i] * F.mse_loss(score, targets[:, i])

    # 2. Feature selection regularization
    feature_reg_loss = 0.0
    for weight_type, weights in model_outputs['weights'].items():
        # Encourage sparse but meaningful feature selection
        feature_reg_loss += torch.mean(torch.abs(weights)) * 0.01  # L1 regularization
        # Ensure weights sum to approximately 1
        feature_reg_loss += torch.abs(weights.sum(dim=-1).mean() - 1.0) * 0.1

    # 3. Temporal consistency loss for historical and future weights
    temporal_consistency_loss = 0.0
    if 'historical' in model_outputs['weights'] and 'future' in model_outputs['weights']:
        historical_weights = model_outputs['weights']['historical']
        future_weights = model_outputs['weights']['future']
        
        # Ensure smooth transitions between historical and future weights
        if historical_weights.size(-1) > 1 and future_weights.size(-1) > 1:
            historical_diffs = historical_weights[:, 1:] - historical_weights[:, :-1]
            future_diffs = future_weights[:, 1:] - future_weights[:, :-1]
            
            temporal_consistency_loss = (
                torch.mean(torch.abs(historical_diffs)) + 
                torch.mean(torch.abs(future_diffs))
            ) * 0.1

    # 4. Equity consideration
    equity_loss = 0.0
    if 'equity_score' in model_outputs['objective_outputs']:
        equity_target = torch.ones_like(model_outputs['objective_outputs']['equity_score'])
        equity_loss = F.mse_loss(
            model_outputs['objective_outputs']['equity_score'],
            equity_target
        ) * 0.2

    # Combine all losses
    total_loss = (
        prediction_loss +
        feature_reg_loss +
        temporal_consistency_loss +
        equity_loss
    )

    return total_loss

def calculate_objective_correlations(feature_weights, config):
    """
    Calculate correlations between objective-specific feature weights.
    
    Args:
        feature_weights (dict): Feature weights for each objective
        config (dict): Configuration containing feature-objective mappings
        
    Returns:
        torch.Tensor: Correlation loss to encourage appropriate relationships
    """
    correlation_loss = 0.0
    
    # Get feature mappings from config
    feature_objective_mapping = config['feature_objective_mapping']
    
    # Calculate correlations between related objectives
    for obj1 in feature_objective_mapping:
        for obj2 in feature_objective_mapping:
            if obj1 != obj2:
                # Check overlap in features
                features1 = set(feature_objective_mapping[obj1])
                features2 = set(feature_objective_mapping[obj2])
                overlap = features1.intersection(features2)
                
                if overlap:
                    weights1 = feature_weights[obj1]
                    weights2 = feature_weights[obj2]
                    
                    # Encourage similar weights for shared features
                    correlation_loss += F.mse_loss(weights1, weights2) * 0.1
    
    return correlation_loss
