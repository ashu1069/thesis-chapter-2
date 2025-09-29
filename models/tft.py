import torch
import torch.nn as nn
from utils.modules import VariableSelectionNetwork, StaticCovariateEncoder, StaticEnrichmentLayer, TemporalSelfAttentionLayer, PositionwiseFeedforwardLayer, TemporalFusionDecoder, SequenceToSequenceLayer, ObjectiveLayer

class TemporalFusionTransformer(nn.Module):
    '''
    The flow of the model is:

    1. Encode static covariates and generate context vectors
    2. Encode historical and future inputs using the cs context
    3. Combine historical and future inputs
    4. Enrich the combined inputs with static information using the ce context
    5. Apply temporal self-attention to the enriched inputs
    6. Apply position-wise feed-forward to the attended inputs
    7. Process the inputs through the temporal fusion decoder
    8. Project the decoder output to the final output size
    '''
    def __init__(self, config):
        super().__init__()
        
        # Add dimension checks
        assert config['hidden_size'] % config['nhead'] == 0, "hidden_size must be divisible by nhead"
        assert config['hidden_size'] == config['lstm_hidden_size'], "hidden_size must match lstm_hidden_size"
        assert config['hidden_size'] == config['context_size'], "hidden_size must match context_size"
        
        # Initialize components with consistent dimensions
        self.static_covariates_encoder = VariableSelectionNetwork(
            input_sizes=config['static_input_sizes'],
            hidden_size=config['hidden_size'],
            dropout=config['dropout']
        )
        
        self.static_context_encoder = StaticCovariateEncoder(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            output_size=config['hidden_size'],
            dropout=config['dropout']
        )
        
        self.historical_inputs_encoder = VariableSelectionNetwork(
            input_sizes=config['historical_input_sizes'],
            hidden_size=config['hidden_size'],
            dropout=config['dropout']
        )
        
        self.future_inputs_encoder = VariableSelectionNetwork(
            input_sizes=config['future_input_sizes'],
            hidden_size=config['hidden_size'],
            dropout=config['dropout']
        )
        
        self.static_enrichment = StaticEnrichmentLayer(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            context_size=config['hidden_size'],
            dropout=config['dropout']
        )
        
        self.temporal_self_attention = TemporalSelfAttentionLayer(
            d_model=config['hidden_size'],
            n_head=config['nhead'],
            dropout=config['dropout']
        )
        
        self.position_wise_feed_forward = PositionwiseFeedforwardLayer(
            input_size=config['hidden_size'],
            hidden_size=config['ff_hidden_size'],
            dropout=config['dropout']
        )
        
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            lstm_hidden_size=config['lstm_hidden_size'],
            dropout=config['dropout'],
            num_lstm_layers=config['num_lstm_layers'],
            num_attention_layers=config['num_attention_layers']
        )
        
        self.output_layer = ObjectiveLayer(
            input_size=config['hidden_size'],
            output_size=config['output_size']
        )
        
        self.config = config  # Store config

    def forward(self, x):
        # Add shape assertions
        for name, tensor in x['static'].items():
            assert len(tensor.shape) == 2, f"Static input {name} must be 2D [batch, features]"
            
        for name, tensor in x['historical'].items():
            assert len(tensor.shape) == 3, f"Historical input {name} must be 3D [batch, time, features]"
            assert tensor.size(1) == self.config['historical_length'], f"Historical input {name} must have {self.config['historical_length']} time steps"
            
        for name, tensor in x['future'].items():
            assert len(tensor.shape) == 3, f"Future input {name} must be 3D [batch, time, features]"
            assert tensor.size(1) == self.config['forecast_length'], f"Future input {name} must have {self.config['forecast_length']} time steps"
            
        # Encode static covariates
        static_encoder_output, static_weights = self.static_covariates_encoder(x['static'])
        
        # Generate static context vectors
        cs, ce, cc, ch = self.static_context_encoder(static_encoder_output)
        
        # Encode historical inputs
        historical_inputs, historical_weights = self.historical_inputs_encoder(x['historical'], context=cs)
        
        # Encode future inputs
        future_inputs, future_weights = self.future_inputs_encoder(x['future'], context=cs)
        
        # Combine historical and future inputs
        # If encoders returned per-variable dimension, reduce over it
        if historical_inputs.dim() == 4:  # [batch, time, num_vars, hidden]
            historical_inputs = historical_inputs.sum(dim=2)
        if future_inputs.dim() == 4:
            future_inputs = future_inputs.sum(dim=2)
        temporal_inputs = torch.cat([historical_inputs, future_inputs], dim=1)
        
        # Apply static enrichment
        enriched_inputs = self.static_enrichment(temporal_inputs, ce)
        
        # Apply temporal self-attention
        attended_inputs = self.temporal_self_attention(enriched_inputs)
        
        # Apply position-wise feed-forward layer
        processed_inputs = self.position_wise_feed_forward(attended_inputs)
        
        # Prepare initial hidden and cell states
        # Ensure hidden and cell are [layers, batch, hidden]
        num_layers = self.config['num_lstm_layers'] if 'num_lstm_layers' in self.config else self.config.get('num_layers', 1)
        # Collapse potential temporal dimension in ch/cc
        if ch.dim() == 3:
            ch = ch[:, -1, :]
        if cc.dim() == 3:
            cc = cc[:, -1, :]
        if ch.dim() == 1:
            ch = ch.unsqueeze(0)
        if cc.dim() == 1:
            cc = cc.unsqueeze(0)
        encoder_hidden = ch.unsqueeze(0).repeat(num_layers, 1, 1)
        encoder_cell = cc.unsqueeze(0).repeat(num_layers, 1, 1)
        
        # Temporal Fusion Decoder expects (x, future_inputs, static_contexts, encoder_hidden, encoder_cell)
        decoded_output, _, _ = self.temporal_fusion_decoder(
            historical_inputs,  # use historical as encoder input
            future_inputs,      # use future inputs separately
            ce,                 # static context
            encoder_hidden,
            encoder_cell
        )
        
        # Final output layer
        objective_outputs = self.output_layer(decoded_output)
        
        # Prepare tensor for training (stack individual scores)
        # Only use the four main objectives, not equity
        objectives_order = [
            'Maximize Health Impact',
            'Maximize Value for Money',
            'Reinforce Financial Sustainability',
            'Support Countries with the Greatest Needs'
        ]
        objective_tensor = torch.cat([
            objective_outputs['individual_scores'][obj] for obj in objectives_order
        ], dim=1)  # [batch, 4]
        
        return {
            'objective_tensor': objective_tensor,  # for training
            'objective_dict': objective_outputs,   # for interpretability
            'weights': {
                'static': static_weights,
                'historical': historical_weights,
                'future': future_weights
            }
        }