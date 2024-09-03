import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from src.simpletts.entity.config_entity import ModelBuildingConfig
from src.simpletts.components.data_transformation import mask_from_seq_lengths
import pandas as pd
from tqdm import tqdm
import os
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Represents a single encoder block in the Transformer architecture.

    This block consists of self-attention followed by a feedforward network,
    with layer normalization and residual connections.
    """
    
    def __init__(self, config : ModelBuildingConfig):
        """
        Initialize the EncoderBlock with its layers.
        """
        super(EncoderBlock, self).__init__()
        self.config = config
        self.norm_1 = nn.LayerNorm(normalized_shape=self.config.embedding_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim= self.config.embedding_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.norm_2 = nn.LayerNorm(normalized_shape= self.config.embedding_size)
        self.linear_1 = nn.Linear( self.config.embedding_size,  self.config.dim_feedforward)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.linear_2 = nn.Linear(self.config.dim_feedforward, self.config.embedding_size)
        self.dropout_3 = torch.nn.Dropout(0.1)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Forward pass of the EncoderBlock.

        Args:
            x (Tensor): Input tensor
            attn_mask (Tensor, optional): Attention mask
            key_padding_mask (Tensor, optional): Key padding mask

        Returns:
            Tensor: Output tensor after passing through the encoder block
        """
        x_out = self.norm_1(x)
        x_out, _ = self.attn(
            query=x_out, 
            key=x_out, 
            value=x_out,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x_out = self.dropout_1(x_out)
        x = x + x_out    

        x_out = self.norm_2(x) 
        x_out = self.linear_1(x_out)
        x_out = F.relu(x_out)
        x_out = self.dropout_2(x_out)
        x_out = self.linear_2(x_out)
        x_out = self.dropout_3(x_out)
        x = x + x_out
        
        return x

class DecoderBlock(nn.Module):
    """
    Represents a single decoder block in the Transformer architecture.

    This block consists of self-attention, encoder-decoder attention, and a feedforward network,
    with layer normalization and residual connections.
    """

    def __init__(self, config : ModelBuildingConfig):
        """
        Initialize the DecoderBlock with its layers.
        """
        super(DecoderBlock, self).__init__()
        self.config = config
        self.norm_1 = nn.LayerNorm(normalized_shape=self.config.embedding_size)
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=self.config.embedding_size,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.norm_2 = nn.LayerNorm(normalized_shape=self.config.embedding_size)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.config.embedding_size,
            num_heads= self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )    
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.norm_3 = nn.LayerNorm(normalized_shape=self.config.embedding_size)
        self.linear_1 = nn.Linear(self.config.embedding_size, self.config.dim_feedforward)
        self.dropout_3 = torch.nn.Dropout(self.config.dropout)
        self.linear_2 = nn.Linear(self.config.dim_feedforward, self.config.embedding_size)
        self.dropout_4 = torch.nn.Dropout(self.config.dropout)

    def forward(self, x, memory, x_attn_mask=None, x_key_padding_mask=None,
                memory_attn_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the DecoderBlock.

        Args:
            x (Tensor): Input tensor
            memory (Tensor): Encoder output
            x_attn_mask (Tensor, optional): Self-attention mask
            x_key_padding_mask (Tensor, optional): Self-attention key padding mask
            memory_attn_mask (Tensor, optional): Encoder-decoder attention mask
            memory_key_padding_mask (Tensor, optional): Encoder-decoder key padding mask

        Returns:
            Tensor: Output tensor after passing through the decoder block
        """
        x_out, _ = self.self_attn(
            query=x, 
            key=x, 
            value=x,
            attn_mask=x_attn_mask,
            key_padding_mask=x_key_padding_mask
        )
        x_out = self.dropout_1(x_out)
        x = self.norm_1(x + x_out)
         
        x_out, _ = self.attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=memory_attn_mask,
            key_padding_mask=memory_key_padding_mask
        )
        x_out = self.dropout_2(x_out)
        x = self.norm_2(x + x_out)

        x_out = self.linear_1(x)
        x_out = F.relu(x_out)
        x_out = self.dropout_3(x_out)
        x_out = self.linear_2(x_out)
        x_out = self.dropout_4(x_out)
        x = self.norm_3(x + x_out)

        return x
class EncoderPreNet(nn.Module):
    def __init__(self, config: ModelBuildingConfig):
        super(EncoderPreNet, self).__init__()
        self.config = config
        
        self.embedding = nn.Embedding(
            num_embeddings=config.text_num_embeddings,
            embedding_dim=config.encoder_embedding_size
        )
        self.linear_1 = nn.Linear(config.encoder_embedding_size, config.encoder_embedding_size)
        self.linear_2 = nn.Linear(config.encoder_embedding_size, config.embedding_size)
        self.conv_1 = nn.Conv1d(
            config.encoder_embedding_size, 
            config.encoder_embedding_size,
            kernel_size=config.encoder_kernel_size, 
            stride=1,
            padding=int((config.encoder_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_1 = nn.BatchNorm1d(config.encoder_embedding_size)
        self.dropout_1 = nn.Dropout(config.dropout)
        self.conv_2 = nn.Conv1d(
            config.encoder_embedding_size, 
            config.encoder_embedding_size,
            kernel_size=config.encoder_kernel_size, 
            stride=1,
            padding=int((config.encoder_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_2 = nn.BatchNorm1d(config.encoder_embedding_size)
        self.dropout_2 = nn.Dropout(config.dropout)
        self.conv_3 = nn.Conv1d(
            config.encoder_embedding_size, 
            config.encoder_embedding_size,
            kernel_size=config.encoder_kernel_size, 
            stride=1,
            padding=int((config.encoder_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_3 = nn.BatchNorm1d(config.encoder_embedding_size)
        self.dropout_3 = nn.Dropout(config.dropout)

    def forward(self, text):
        """
        Forward pass of the EncoderPreNet.

        Args:
            text (Tensor): Input text tensor

        Returns:
            Tensor: Processed text tensor
        """
        x = self.embedding(text) # (N, S, E)
        x = self.linear_1(x)
        x = x.transpose(2, 1) # (N, E, S) 
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)    
        x = F.relu(x)
        x = self.dropout_3(x)
        x = x.transpose(1, 2) # (N, S, E)
        x = self.linear_2(x)
        return x
    
class PostNet(nn.Module):
    """
    Post-network that refines the output of the decoder.

    This network applies a series of convolutional layers to the mel spectrogram.
    """

    def __init__(self, config : ModelBuildingConfig):
        """
        Initialize the PostNet with its layers.
        """
        super(PostNet, self).__init__()  
        self.config = config
        self.conv_1 = nn.Conv1d(
            self.config.mel_freq, 
            self.config.postnet_embedding_size,
            kernel_size=self.config.postnet_kernel_size, 
            stride=1,
            padding=int((self.config.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_1 = nn.BatchNorm1d(self.config.postnet_embedding_size)
        self.dropout_1 = torch.nn.Dropout(0.5)
        self.conv_2 = nn.Conv1d(
            self.config.postnet_embedding_size, 
            self.config.postnet_embedding_size,
            kernel_size=self.config.postnet_kernel_size, 
            stride=1,
            padding=int((self.config.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_2 = nn.BatchNorm1d(self.config.postnet_embedding_size)
        self.dropout_2 = torch.nn.Dropout(0.5)
        self.conv_3 = nn.Conv1d(
            self.config.postnet_embedding_size, 
            self.config.postnet_embedding_size,
            kernel_size=self.config.postnet_kernel_size, 
            stride=1,
            padding=int((self.config.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_3 = nn.BatchNorm1d(self.config.postnet_embedding_size)
        self.dropout_3 = torch.nn.Dropout(0.5)
        self.conv_4 = nn.Conv1d(
            self.config.postnet_embedding_size, 
            self.config.postnet_embedding_size,
            kernel_size=self.config.postnet_kernel_size, 
            stride=1,
            padding=int((self.config.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_4 = nn.BatchNorm1d(self.config.postnet_embedding_size)
        self.dropout_4 = torch.nn.Dropout(0.5)
        self.conv_5 = nn.Conv1d(
            self.config.postnet_embedding_size, 
            self.config.postnet_embedding_size,
            kernel_size=self.config.postnet_kernel_size, 
            stride=1,
            padding=int((self.config.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_5 = nn.BatchNorm1d(self.config.postnet_embedding_size)
        self.dropout_5 = torch.nn.Dropout(0.5)
        self.conv_6 = nn.Conv1d(
            self.config.postnet_embedding_size, 
            self.config.mel_freq,
            kernel_size=self.config.postnet_kernel_size, 
            stride=1,
            padding=int((self.config.postnet_kernel_size - 1) / 2), 
            dilation=1
        )
        self.bn_6 = nn.BatchNorm1d(self.config.mel_freq)
        self.dropout_6 = torch.nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass of the PostNet.

        Args:
            x (Tensor): Input mel spectrogram

        Returns:
            Tensor: Refined mel spectrogram
        """
        x = x.transpose(2, 1) # (N, FREQ, TIME)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = torch.tanh(x)
        x = self.dropout_1(x) # (N, POSNET_DIM, TIME)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = torch.tanh(x)
        x = self.dropout_2(x) # (N, POSNET_DIM, TIME)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = torch.tanh(x)
        x = self.dropout_3(x) # (N, POSNET_DIM, TIME)    
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = torch.tanh(x)
        x = self.dropout_4(x) # (N, POSNET_DIM, TIME)    
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = torch.tanh(x)
        x = self.dropout_5(x) # (N, POSNET_DIM, TIME)
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.dropout_6(x) # (N, FREQ, TIME)
        x = x.transpose(1, 2)
        return x

class DecoderPreNet(nn.Module):
    def __init__(self, config : ModelBuildingConfig):
        super(DecoderPreNet, self).__init__()
        self.config = config 
        self.linear_1 = nn.Linear(self.config.mel_freq, self.config.embedding_size)
        self.linear_2 = nn.Linear(self.config.embedding_size, self.config.embedding_size)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)
        x = self.linear_2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)
        return x
        
class DecoderPreNet(nn.Module):
    """
    Decoder pre-network that processes mel spectrograms before the main decoder.

    This network applies linear transformations with dropout.
    """

    def __init__(self, config : ModelBuildingConfig):
        """
        Initialize the DecoderPreNet with its layers.
        """
        super(DecoderPreNet, self).__init__()
        self.config = config
        self.linear_1 = nn.Linear(self.config.mel_freq, self.config.embedding_size)
        self.linear_2 = nn.Linear(self.config.embedding_size, self.config.embedding_size)

    def forward(self, x):
        """
        Forward pass of the DecoderPreNet.

        Args:
            x (Tensor): Input mel spectrogram

        Returns:
            Tensor: Processed mel spectrogram
        """
        x = self.linear_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)
        x = self.linear_2(x)
        x = F.relu(x)    
        x = F.dropout(x, p=0.5, training=True)
        return x
    
from src.simpletts.components.data_transformation import mask_from_seq_lengths
class TransformerTTS(nn.Module):
    def __init__(self, config: ModelBuildingConfig, device: str = "cuda"):
        super(TransformerTTS, self).__init__()
        self.config = config
        self.device = device

        # Changed: Using config parameter for all submodule initializations
        self.encoder_prenet = EncoderPreNet(config)
        self.decoder_prenet = DecoderPreNet(config)
        self.postnet = PostNet(config)
        self.pos_encoding = nn.Embedding(config.max_mel_time, config.embedding_size)
        self.encoder_block_1 = EncoderBlock(config)
        self.encoder_block_2 = EncoderBlock(config)
        self.encoder_block_3 = EncoderBlock(config)
        self.decoder_block_1 = DecoderBlock(config)
        self.decoder_block_2 = DecoderBlock(config)
        self.decoder_block_3 = DecoderBlock(config)
        self.linear_1 = nn.Linear(config.embedding_size, config.mel_freq)
        self.linear_2 = nn.Linear(config.embedding_size, 1)
        self.norm_memory = nn.LayerNorm(config.embedding_size)

    def forward(self, text, text_len, mel, mel_len):
        """
        Forward pass of the TransformerTTS model.

        Args:
            text (Tensor): Input text tensor
            text_len (Tensor): Lengths of input texts
            mel (Tensor): Target mel spectrogram
            mel_len (Tensor): Lengths of target mel spectrograms

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Predicted mel spectrogram (post-net), 
                                           predicted mel spectrogram (pre-net),
                                           stop token predictions
        """
        N = text.shape[0]
        S = text.shape[1]
        TIME = mel.shape[1]

        # Create masks
        self.src_key_padding_mask = torch.zeros((N, S), device=text.device).masked_fill(
            ~mask_from_seq_lengths(text_len, max_length=S), float("-inf")
        )
        self.src_mask = torch.zeros((S, S), device=text.device).masked_fill(
            torch.triu(torch.full((S, S), True, dtype=torch.bool), diagonal=1).to(text.device),       
            float("-inf")
        )
        self.tgt_key_padding_mask = torch.zeros((N, TIME), device=mel.device).masked_fill(
            ~mask_from_seq_lengths(mel_len, max_length=TIME), float("-inf")
        )
        self.tgt_mask = torch.zeros((TIME, TIME), device=mel.device).masked_fill(
            torch.triu(torch.full((TIME, TIME), True, device=mel.device, dtype=torch.bool), diagonal=1),       
            float("-inf")
        )
        self.memory_mask = torch.zeros((TIME, S), device=mel.device).masked_fill(
            torch.triu(torch.full((TIME, S), True, device=mel.device, dtype=torch.bool), diagonal=1),       
            float("-inf")
        )    

        # Encoder
        text_x = self.encoder_prenet(text)
        pos_codes = self.pos_encoding(torch.arange(self.config.max_mel_time).to(mel.device))
        S = text_x.shape[1]
        text_x = text_x + pos_codes[:S]
        text_x = self.encoder_block_1(text_x, attn_mask=self.src_mask, key_padding_mask=self.src_key_padding_mask)
        text_x = self.encoder_block_2(text_x, attn_mask=self.src_mask, key_padding_mask=self.src_key_padding_mask)
        text_x = self.encoder_block_3(text_x, attn_mask=self.src_mask, key_padding_mask=self.src_key_padding_mask)
        text_x = self.norm_memory(text_x)
        
        # Decoder
        mel_x = self.decoder_prenet(mel)
        mel_x = mel_x + pos_codes[:TIME]
        mel_x = self.decoder_block_1(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask, 
                                     x_key_padding_mask=self.tgt_key_padding_mask,
                                     memory_attn_mask=self.memory_mask,
                                     memory_key_padding_mask=self.src_key_padding_mask)
        mel_x = self.decoder_block_2(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask, 
                                     x_key_padding_mask=self.tgt_key_padding_mask,
                                     memory_attn_mask=self.memory_mask,
                                     memory_key_padding_mask=self.src_key_padding_mask)
        mel_x = self.decoder_block_3(x=mel_x, memory=text_x, x_attn_mask=self.tgt_mask, 
                                     x_key_padding_mask=self.tgt_key_padding_mask,
                                     memory_attn_mask=self.memory_mask,
                                     memory_key_padding_mask=self.src_key_padding_mask)

        # Output processing
        mel_linear = self.linear_1(mel_x)
        mel_postnet = self.postnet(mel_linear)
        mel_postnet = mel_linear + mel_postnet
        stop_token = self.linear_2(mel_x)

        # Masking
        bool_mel_mask = self.tgt_key_padding_mask.ne(0).unsqueeze(-1).repeat(1, 1, self.config.mel_freq)
        mel_linear = mel_linear.masked_fill(bool_mel_mask, 0)
        mel_postnet = mel_postnet.masked_fill(bool_mel_mask, 0)
        stop_token = stop_token.masked_fill(bool_mel_mask[:, :, 0].unsqueeze(-1), 1e3).squeeze(2)
        
        return mel_postnet, mel_linear, stop_token 
    
class BuildModel:
    def __init__(self, config: ModelBuildingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build(self):
        model = TransformerTTS(config=self.config, device=self.device)
        return model

    def save_model(self, model: nn.Module, file_name: str = "model.pth"):
        root_dir = str(self.config.root_dir)
        save_path = os.path.join(root_dir, file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def build_and_save(self):
        model = self.build()
        self.save_model(model)
        return model