import torch
import torch.nn as nn
import math
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self attention
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2.transpose(0, 1), 
            tgt2.transpose(0, 1), 
            tgt2.transpose(0, 1),
            attn_mask=tgt_mask
        )
        tgt = tgt + self.dropout1(tgt2.transpose(0, 1))
        
        # Cross attention
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            query=tgt2.transpose(0, 1),
            key=memory.transpose(0, 1),
            value=memory.transpose(0, 1),
            attn_mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2.transpose(0, 1))
        
        # Feed forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class DecoderWithTransformer(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=2048, dropout=0.5, nhead=8):
        super().__init__()
        self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # Transformer components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=decoder_dim*4,
            dropout=dropout
        )
        self.feature_proj = nn.Linear(features_dim, decoder_dim)
        self.output_proj = nn.Linear(decoder_dim, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.output_proj.bias.data.fill_(0)
        self.output_proj.weight.data.uniform_(-0.1, 0.1)

    def forward(self, image_features, encoded_captions, caption_lengths):
        batch_size = image_features.size(0)
        
        # Sort input data
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding with positional encoding
        embeddings = self.embedding(encoded_captions)  # (batch_size, seq_len, embed_dim)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.dropout_layer(embeddings)
        
        # Project image features
        memory = self.feature_proj(image_features)  # (batch_size, 36, decoder_dim)
        
        # Transformer decoding
        tgt = embeddings
        for _ in range(3):  # 3 transformer layers
            tgt = self.transformer_layer(tgt, memory)
        
        # Output projections
        output = self.output_proj(tgt)
        
        # Create predictions tensor
        predictions = torch.zeros(batch_size, max(caption_lengths)-1, self.vocab_size).to(device)
        for t in range(max(caption_lengths)-1):
            batch_size_t = sum([l > t for l in caption_lengths])
            predictions[:batch_size_t, t, :] = output[:batch_size_t, t, :]
        
        return predictions, None, encoded_captions, [l-1 for l in caption_lengths], sort_ind
