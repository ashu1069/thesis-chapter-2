import torch
import torch.nn.functional as F

def custom_loss(model_outputs, targets, objective_weights):
    """
    Args:
        model_outputs (dict): Contains 'objective_outputs' tensor of shape [batch_size, num_objectives]
        targets (tensor): Shape [batch_size, num_objectives]
        objective_weights (list): Weights for each objective
    """
    assert len(objective_weights) == targets.size(1), "Number of weights must match number of objectives"
    
    # Convert weights to tensor
    weights = torch.tensor(objective_weights, device=targets.device)
    
    # Ensure model outputs and targets have same shape
    outputs = model_outputs['objective_outputs']
    assert outputs.shape == targets.shape, f"Output shape {outputs.shape} must match target shape {targets.shape}"
    
    # Calculate weighted MSE loss
    loss = torch.sum(weights * torch.mean((outputs - targets) ** 2, dim=0))
    
    return loss

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
