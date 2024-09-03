from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_dir : Path
    local_data_file: Path
    unzip_dir : Path
    
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    wave_path : Path
    csv_path : Path
    sr  : int
    n_fft : int
    n_stft : int
    frame_length : float
    win_length : int
    mel_freq : int
    max_mel_time : int
    max_db : int
    scale_db : int
    ref : float
    power : float
    norm_db : int
    ampl_multiplier : float
    ampl_amin : str
    db_multiplier : float
    ampl_ref : float
    ampl_power : float
    min_level_db : float
    frame_shift : float
    hop_length : int
    
    
@dataclass(frozen=True)
class ModelBuildingConfig:
    root_dir : Path
    text_num_embeddings : int
    embedding_size : int
    encoder_embedding_size  : int
    dim_feedforward : int
    postnet_embedding_size : int
    encoder_kernel_size : int
    postnet_kernel_size : int
    num_heads : int
    dropout : float
    batch_first : bool
    mel_freq : int
    max_mel_time : int
    
    