import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Union

class TimeDistributed(nn.Module):
    '''
    A PyTorch module to apply a given module independently at each time step of a sequence.
    batch_first: indicates if batch dimension is the first dimension in the input tensor.
    '''
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        '''
        if the input tensor has only two dimensions or less, it directly applies the module to x, 
        otherwise it reshapes x to combine the batch and time steps into one dimentions and applies
        the module to the reshaped tensor. One also has to reshape the output back to original to the 
        original dimensions, taking into account whether the first dimension is batch or not.
        '''
        if len(x.size()) <=2:
            return self.module(x)
        
        #squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1)) # (samples*timesteps, input_size)

        y = self.module(x_reshape)

        #reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1)) # (timesteps, samples, output_size)

        return y
    
class TimeDistributedInterpolation(nn.Module):
    '''
    A PyTorch module designed to apply interpolation independently to each time step of a sequence. 
    This class is particularly useful when you need to upsample or interpolate features for each time step in a sequence. 
    It also includes an option to make the interpolation trainable, which allows the model to learn how to scale the interpolated values.
    '''
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()
    
    def interpolate(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 2:
            # Input shape: (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
            upsampled = F.interpolate(x, size=self.output_size, mode="linear", align_corners=True)
            
        elif len(orig_shape) == 3:
            # Input shape: (batch, sequence, features)
            batch_size, seq_len, features = orig_shape
            # Reshape to (batch * seq_len, 1, features)
            x_reshaped = x.reshape(-1, 1, features)
            # Interpolate
            upsampled = F.interpolate(x_reshaped, size=self.output_size, mode="linear", align_corners=True)
            # Reshape back to (batch, sequence, output_size)
            upsampled = upsampled.reshape(batch_size, seq_len, self.output_size)
            
        elif len(orig_shape) == 4:
            # Input shape: (batch, channels, sequence, features)
            batch_size, channels, seq_len, features = orig_shape
            # Reshape to (batch * channels * seq_len, 1, features)
            x_reshaped = x.reshape(-1, 1, features)
            # Interpolate
            upsampled = F.interpolate(x_reshaped, size=self.output_size, mode="linear", align_corners=True)
            # Reshape back to (batch, channels, sequence, output_size)
            upsampled = upsampled.reshape(batch_size, channels, seq_len, self.output_size)
        
        if self.trainable:
            if len(upsampled.shape) == 4:
                mask = self.gate(self.mask) * 2.0
                upsampled = upsampled * mask.view(1, 1, 1, -1)
            else:
                upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        
        return upsampled
    
    def forward(self, x):
        return self.interpolate(x)

class GatedLinearUnit(nn.Module):
    '''
    A neural network module that combines a linear transformation with a gating mechanism. 
    It allows for more expressive power compared to a simple linear layer by learning gates that control which parts of the input pass through.

    Dropout: applies to the input to prevent overfitting
    Linear: projects the input to twice the hidden size which allows the layer to learn both the transformation and the gate.
    GLU: splits the input into two parts- one part undergoes a linear transformation, while the other part serves as a gate, 
         which controls the output by modulating the transformed input.
    '''
    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x
    
class ResampleNorm(nn.Module):
    '''
    Resampling: resample the input tensor to desired output tensor
    Trainable Mask: If enabled, the training mask allows the model to learn how to scale the resampled data dynamically.
    Normalization: to ensure that the output has a consistent scale and zero mean, which can help stabilize training and performance
    '''
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.trainable_add = trainable_add

        # If the input size is different from the output size, we will use our interpolation method to handle the resampling
        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output
    
class AddNorm(nn.Module):
    '''
    Designed to add a skip connection from a previous layer to the current input tensor x and then normalize the result.
    '''
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = nn.Linear(self.skip_size, self.input_size)
        # Add projection for runtime shape mismatch
        self.dynamic_proj = None
        
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # Handle feature dimension mismatch
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        # Handle sequence length mismatch only for 3D tensors
        if len(x.shape) == 3 and len(skip.shape) == 3 and x.size(1) != skip.size(1):
            x = F.interpolate(
                x.transpose(1, 2),
                size=skip.size(1),
                mode='linear',
                align_corners=True
            ).transpose(1, 2)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        # Project skip at runtime if feature dims still don't match
        if skip.shape[-1] != x.shape[-1]:
            if not hasattr(self, '_runtime_proj') or self._runtime_proj.in_features != skip.shape[-1] or self._runtime_proj.out_features != x.shape[-1]:
                self._runtime_proj = nn.Linear(skip.shape[-1], x.shape[-1]).to(skip.device)
            skip = self._runtime_proj(skip)
            # Re-initialize LayerNorm if feature size changed
            if self.norm.normalized_shape != (x.shape[-1],):
                self.norm = nn.LayerNorm(x.shape[-1]).to(x.device)

        output = self.norm(x + skip)
        return output
    
class GateAddNorm(nn.Module):
    '''
    Designed to combine the functionalities of a Gated Linear Unit (GLU) and an AddNorm layer.
    This class is useful for creating a block that applies a GLU for non-linear transformation followed 
    by adding a skip connection and normalizing the result, a pattern often used in transformer architectures.
    '''
    def __init__(
            self,
            input_size: int,
            hidden_size: int = None,
            skip_size: int = None,
            trainable_add: bool = False,
            dropout: float = None
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.trainable_add = trainable_add
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output

class GatedResidualNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.1,
            context_size: int = None,
            residual: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        # Ensure consistent dimensions throughout the network
        if self.input_size != self.output_size and not self.residual:
            self.resample_norm = ResampleNorm(self.input_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.output_size,
            hidden_size=self.output_size,
            skip_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        # Handle dimension differences
        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        # Main network operations
        x = self.fc1(x)
        if hasattr(self, 'context') and context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)

        # Handle sequence length differences before gate_norm
        if len(x.shape) == 3 and len(residual.shape) == 3 and x.size(1) != residual.size(1):
            # Only interpolate if dealing with 3D tensors (temporal data)
            x = F.interpolate(
                x.transpose(1, 2),
                size=residual.size(1),
                mode='linear',
                align_corners=True
            ).transpose(1, 2)

        x = self.gate_norm(x, residual)
        return x
    
class VariableSelectionNetwork(nn.Module):
    '''
    Designed to dynamically select relevant variables and transform them for further processing in the Temporal Fusion Transformer (TFT). 
    This class leverages gated residual networks (GRNs) to process individual variables and context information.
    '''
    def __init__(
            self, 
            input_sizes: Dict[str, int],    # dictionary of variable names and their respective input size
            hidden_size: int,   # hidden size of the GRNs
            input_embedding_flags: Dict[str, bool] = {},    # flags indicating which input requires embedding
            dropout: float = 0.1,   # dropout rate
            context_size: int = None,   # size of the context vector (optional)
            single_variable_grns: Dict[str, GatedResidualNetwork] = {}, # predefined GRNs for individual variables (optional)
            prescalers: Dict[str, nn.Linear] = {}   # predefined linear layers for scaling inputs (optional)

    ):
        '''
        calculate weights for "num_inputs" variables which are each of size "input_size"
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        # It sets up a GatedResidualNetwork for combining multiple variables if there are more than one.
        if self.num_inputs > 1:
            # Ensure output_size and skip_size match num_inputs for flattened_grn
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    self.num_inputs,  # hidden_size
                    self.num_inputs,  # output_size
                    self.dropout,
                    self.context_size,
                    residual=False
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    self.num_inputs,  # hidden_size
                    self.num_inputs,  # output_size
                    self.dropout,
                    residual=False
                )

        # It sets up GRNs and prescalers for each variable. If a variable requires embedding, it uses ResampleNorm; otherwise, it uses a GRN.
        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout = self.dropout
                )
            if name in prescalers:
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)

        # Initializes the softmax function for calculating variable weights.
        self.softmax = nn.Softmax(dim=1)

    @property
    def input_size_total(self): # Sum of input sizes for all variables.
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())
    
    @property
    def num_inputs(self): # number of input variables
        return len(self.input_sizes)
    
    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # Transform single variables
            var_outputs = []
            weight_inputs = []
            
            for name in self.input_sizes.keys():
                # Select embedding belonging to single point
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                processed_var = self.single_variable_grns[name](variable_embedding)
                var_outputs.append(processed_var)
            
            # Stack and process variable outputs
            # If temporal (3D), stack along variable axis to get [batch, time, num_vars, hidden_size]
            if var_outputs[0].dim() == 3:
                var_outputs = torch.stack(var_outputs, dim=2)  # [batch, time, num_vars, hidden_size]
            else:
                var_outputs = torch.stack(var_outputs, dim=1)  # [batch, num_vars, hidden_size]
            
            # Calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights)  # [batch_size, num_vars]
            
            # Add necessary dimensions to match var_outputs
            if len(var_outputs.shape) == 4:  # For temporal inputs [batch, seq_len, num_vars, hidden_size]
                # Reshape sparse_weights to [batch_size, 1, num_vars, 1]
                sparse_weights = sparse_weights.unsqueeze(1).unsqueeze(-1)
                # Weighted sum over num_vars
                outputs = (var_outputs * sparse_weights).sum(dim=2)  # [batch, seq_len, hidden_size]
            else:  # For static inputs [batch, num_vars, hidden_size]
                # Reshape sparse_weights to [batch_size, num_vars, 1]
                sparse_weights = sparse_weights.unsqueeze(-1)
                # Weighted sum over num_vars
                outputs = (var_outputs * sparse_weights).sum(dim=1)  # [batch, hidden_size]
                outputs = outputs.unsqueeze(1)  # [batch, 1, hidden_size] for consistency
        else:
            # For one input, don't perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)
            # Only unsqueeze for static data (2D to 3D)
            if outputs.dim() == 2 and variable_embedding.dim() == 2:
                outputs = outputs.unsqueeze(1)  # [batch, 1, hidden_size]
            sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)

        return outputs, sparse_weights

class PositionalEncoder(nn.Module):
    '''
    Designed to add positional information to the input embeddings, which is essential for transformer models to capture the order of sequences. 
    This class implements sinusoidal positional encoding as described in the original Transformer paper "Attention is All You Need."
    '''
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "Model dimension has to be a multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        # The positional encodings are precomputed and stored in a tensor (pe) with dimensions [max_seq_len, d_model].
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))

        pe = pe.unsqueeze(0)

        # The precomputed positional encodings are registered as a buffer (self.pe) so that they are not updated during training.
        self.register_buffer("pe", pe)

    def forward(self, x):
        '''
        It scales the input by the square root of the model dimension to balance the influence of the positional encodings,
        and ensures that the positional encodings are added to the input tensor without requiring gradients.
        '''
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len, :]
            x = x + pe
            return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        '''
        1. Computes the attention scores using the query (q), key (k), and value (v) matrices.
        2. Computes the dot product between the query and key matrices.
        3. If scaling is enabled, scales the attention scores by the square root of the dimensionality of the key matrix.
        4. Applies an optional mask to the attention scores.
        5. Applies the softmax function to obtain the attention weights.
        6. Applies dropout to the attention weights if specified.
        7. Computes the output as the dot product of the attention weights and the value matrix.
        8. Returns the output and the attention weights.
        '''
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = math.sqrt(k.size(-1))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        '''
        1. Computes the dimensionality of the keys (d_k), queries (d_q), and values (d_v).
        2. Initializes linear layers for projecting values (v_layer), queries (q_layers), and keys (k_layers).
        3. Initializes a ScaledDotProductAttention module for computing attention scores.
        4. Initializes a linear layer (w_h) for combining the output from all attention heads.
        5. Initializes weights using the Xavier uniform initialization.
        '''
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        1. The forward method computes the attention scores and outputs for the given queries (q), keys (k), and values (v).
        2. Projects the values using the v_layer.
        3. For each attention head, projects the queries and keys using their respective linear layers, computes the attention scores and outputs using the ScaledDotProductAttention module, and applies dropout.
        4. Stacks the attention outputs and attention scores from all heads.
        5. Averages the attention outputs across all heads if there is more than one head.
        6. Combines the outputs from all heads using the w_h linear layer and applies dropout.
        '''
        heads = []
        attns = []
        vs = self.v_layer(v)

        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head = self.dropout(head)
            heads.append(head)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn

class StaticCovariateEncoder(nn.Module):
    '''
    Encodes static features to produce four different context vectors using separate GRN encoders.
    '''
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.grn_cs = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        )
        self.grn_ce = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        )
        self.grn_cc = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        )
        self.grn_ch = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout
        )

    def forward(self, x):
        cs = self.grn_cs(x)
        ce = self.grn_ce(x)
        cc = self.grn_cc(x)
        ch = self.grn_ch(x)
        return cs, ce, cc, ch

class SequenceToSequenceLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size)  # Transform input to hidden size

    def forward(self, x, future_inputs, encoder_hidden, encoder_cell):
        batch_size, seq_len, _ = x.shape
        future_len = future_inputs.shape[1]
        
        # Transform input size to hidden size
        x_transformed = self.fc(x)
        
        # Encoder
        _, (hidden, cell) = self.lstm_encoder(x_transformed, (encoder_hidden, encoder_cell))
        
        # Decoder
        decoder_input = future_inputs
        decoder_outputs = []
        for t in range(future_len):
            decoder_output, (hidden, cell) = self.lstm_decoder(decoder_input[:, t:t+1, :], (hidden, cell))
            decoder_outputs.append(decoder_output)
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        return decoder_outputs, hidden, cell

class StaticEnrichmentLayer(nn.Module):
    def __init__(self, input_size, hidden_size, context_size, dropout):
        super(StaticEnrichmentLayer, self).__init__()
        self.grn = GatedResidualNetwork(input_size, hidden_size, input_size, dropout, context_size)

    def forward(self, x, context):
        return self.grn(x, context)

class TemporalSelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TemporalSelfAttentionLayer, self).__init__()
        self.attention = InterpretableMultiHeadAttention(n_head, d_model, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.glu = GatedLinearUnit(d_model, d_model, dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        attn_output = self.layer_norm(x + attn_output)
        attn_output = self.glu(attn_output)
        return attn_output

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.grn = GatedResidualNetwork(input_size, hidden_size, input_size, dropout)

    def forward(self, x):
        return self.grn(x)

class TemporalFusionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_hidden_size, dropout, num_lstm_layers, num_attention_layers):
        super().__init__()
        self.seq2seq = SequenceToSequenceLayer(input_size, lstm_hidden_size, num_lstm_layers)
        self.static_enrichment = StaticEnrichmentLayer(hidden_size, hidden_size, hidden_size, dropout)  # context_size = hidden_size
        self.self_attention = TemporalSelfAttentionLayer(hidden_size, num_attention_layers, dropout)
        self.feedforward = PositionwiseFeedforwardLayer(hidden_size, hidden_size, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.glu = GatedLinearUnit(hidden_size, hidden_size, dropout)

    def forward(self, x, future_inputs, static_contexts, encoder_hidden, encoder_cell, mask=None):
        # Sequence-to-Sequence Layer
        seq_output, hidden, cell = self.seq2seq(x, future_inputs, encoder_hidden, encoder_cell)
        seq_output = self.layer_norm(x + seq_output)
        seq_output = self.glu(seq_output)

        # Static Enrichment Layer
        enriched_output = self.static_enrichment(seq_output, static_contexts)

        # Temporal Self-Attention Layer
        attn_output = self.self_attention(enriched_output, mask)

        # Position-wise Feed-forward Layer
        feedforward_output = self.feedforward(attn_output)

        return feedforward_output, hidden, cell

class ObjectiveLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int = 5):
        super().__init__()
        
        # Define objectives and their weights
        self.objectives = {
            'Maximize Health Impact': 0.30,
            'Maximize Value for Money': 0.30,
            'Reinforce Financial Sustainability': 0.25,
            'Support Countries with the Greatest Needs': 0.15,
            'Promote Equitable Distribution': None  # Applied across all objectives
        }

        # Define feature mappings for each objective
        self.feature_mappings = {
            'Maximize Health Impact': [
                'Endemic Potential R0', 'Endemic Potential CFR', 'Endemic Potential Duration',
                'Healthcare Index Tier-X hospitals', 'Healthcare Index Workforce capacity',
                'Healthcare Index Bed availability per capita', 'Immunization Coverage',
                'Frequency of outbreaks', 'Magnitude of outbreaks Deaths',
                'Magnitude of outbreaks Infected', 'Magnitude of outbreaks Severity Index'
            ],
            'Maximize Value for Money': [
                'Economic Index Budget allocation per capita', 'Economic Index Fraction of total budget',
                'Healthcare Index Expenditure per capita', 'Socio-economic GDP per capita',
                'Socio-economic Employment Rates', 'Socio-economic Education Levels'
            ],
            'Reinforce Financial Sustainability': [
                'Economic Index Budget allocation per capita', 'Economic Index Fraction of total budget',
                'Healthcare Index Expenditure per capita', 'Socio-economic GDP per capita',
                'Socio-economic Employment Rates', 'Socio-economic Poverty Rates',
                'Political Stability Index', 'Communication Affordability'
            ],
            'Support Countries with the Greatest Needs': [
                'Demography Urban-Rural Split', 'Demography Population Density', 'Environmental Index',
                'Socio-economic Gini Index', 'Socio-economic Poverty Rates',
                'Healthcare Index Bed availability per capita', 'Political Stability Index',
                'Security and Conflict Index'
            ],
            'Promote Equitable Distribution': []  # Will use all features
        }

        # Initialize feature weights for each objective
        self.feature_weights = nn.ParameterDict({
            objective: nn.Parameter(torch.rand(len(features))) 
            for objective, features in self.feature_mappings.items()
            if len(features) > 0  # Skip empty feature lists
        })

        # Create objective-specific processing networks
        self.objective_heads = nn.ModuleDict({
            objective: nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.ReLU(),
                nn.LayerNorm(input_size // 2),
                nn.Dropout(0.1),
                nn.Linear(input_size // 2, 1),
                nn.Sigmoid()
            ) for objective in self.objectives.keys()
        })

    def forward(self, x):
        # Use the last timestep's output for prediction
        last_hidden = x[:, -1, :]  # [batch_size, input_size]
        
        # Calculate scores for each objective
        objective_scores = {}
        weighted_scores = {}
        
        # First calculate equity score as it affects all other objectives
        equity_score = self.objective_heads['Promote Equitable Distribution'](last_hidden)
        
        # Calculate scores for other objectives
        for objective, weight in self.objectives.items():
            if objective != 'Promote Equitable Distribution':
                # Apply feature-specific weights if available
                if objective in self.feature_weights:
                    feature_weights = F.softmax(self.feature_weights[objective], dim=0)
                    weighted_features = last_hidden * feature_weights.unsqueeze(0)
                else:
                    weighted_features = last_hidden
                
                # Calculate base score
                score = self.objective_heads[objective](weighted_features)
                
                # Apply equity consideration
                score = score * equity_score
                
                # Store scores
                objective_scores[objective] = score
                if weight is not None:  # Apply objective weight if exists
                    weighted_scores[objective] = score * weight

        # Calculate final weighted score
        final_score = sum(weighted_scores.values())
        
        return {
            'individual_scores': objective_scores,
            'equity_score': equity_score,
            'weighted_scores': weighted_scores,
            'final_score': final_score
        }

    def get_feature_importance(self):
        """Returns the learned feature importance weights for each objective"""
        return {
            objective: F.softmax(weights, dim=0).detach()
            for objective, weights in self.feature_weights.items()
        }