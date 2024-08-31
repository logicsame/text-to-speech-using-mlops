import os
import torch
import torchaudio
from torchaudio.functional import spectrogram
from src.simpletts.entity.config_entity import DataTransformationConfig
from src.simpletts.logging import logger
import pandas as pd
from sklearn.model_selection import train_test_split
symbols = [
    'EOS', ' ', '!', ',', '-', '.', \
    ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
    'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
  ]


symbol_to_id = {
  s: i for i, s in enumerate(symbols)
}

def mask_from_seq_lengths(
    sequence_lengths: torch.Tensor, 
    max_length: int
) -> torch.BoolTensor:
   
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor 
  
def text_to_seq(text):
    text = text.lower()
    seq = []
    for s in text:
        _id = symbol_to_id.get(s, None)
        if _id is not None:
            seq.append(_id)
    seq.append(symbol_to_id['EOS'])
    return torch.IntTensor(seq)



class AudioProcessor:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
        print(type(self.config.min_level_db))
        print(type(self.config.max_db))
        print(type(self.config.norm_db))
        print(type(self.config.ref))

        
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            power=self.config.power
        )
        
        self.mel_scale_transform = torchaudio.transforms.MelScale(
            n_mels=self.config.mel_freq,
            sample_rate=self.config.sr,
            n_stft=self.config.n_stft
        )
        
        self.mel_inverse_transform = torchaudio.transforms.InverseMelScale(
            n_mels=self.config.mel_freq,
            sample_rate=self.config.sr,
            n_stft=self.config.n_stft
        ).cuda()
        
        self.griffnlim_transform = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length
        ).cuda()
        
    def norm_mel_spec_db(mel_spec): 
        min_level_db = -100.0
        max_db = 100
        norm_db = 10
        ref = 4.0
        mel_spec = ((2.0*mel_spec - min_level_db) / (max_db/norm_db)) - 1.0
        mel_spec = torch.clip(mel_spec, -ref*norm_db, ref*norm_db)
        return mel_spec


    
    def denorm_mel_spec_db(self, mel_spec):
        mel_spec = (((1.0 + mel_spec) * (self.config.max_db / self.config.norm_db)) + self.config.min_level_db) / 2.0
        return mel_spec
    
    def pow_to_db_mel_spec(self, mel_spec):
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier=self.config.ampl_multiplier,
            amin=self.config.ampl_amin,
            db_multiplier=self.config.db_multiplier,
            top_db=self.config.max_db
        )
        mel_spec = mel_spec / self.config.scale_db
        return mel_spec
    
    def db_to_power_mel_spec(self, mel_spec):
        mel_spec = mel_spec * self.config.scale_db
        mel_spec = torchaudio.functional.DB_to_amplitude(
            mel_spec,
            ref=self.config.ampl_ref,
            power=self.config.ampl_power
        )
        return mel_spec
    
    def convert_to_mel_spec(self, wav):
        spec = self.spec_transform(wav)
        mel_spec = self.mel_scale_transform(spec)
        db_mel_spec = self.pow_to_db_mel_spec(mel_spec)
        db_mel_spec = db_mel_spec.squeeze(0)
        return db_mel_spec
    
    def inverse_mel_spec_to_wav(self, mel_spec):
        power_mel_spec = self.db_to_power_mel_spec(mel_spec)
        spectrogram = self.mel_inverse_transform(power_mel_spec)
        pseudo_wav = self.griffnlim_transform(spectrogram)
        return pseudo_wav


import torch.utils


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, df, config: DataTransformationConfig):
        self.df = df
        self.cache = {}
        self.config = config
        self.audio_processor = AudioProcessor(config)  # Pass the config here
        
    def get_item(self, row):
        wav_id = row['wav']
        wav_path = f"{self.config.wave_path}/{wav_id}.wav"
        text = row['text_norm']
        text = text_to_seq(text)
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        assert sample_rate == self.config.sr
        mel = self.audio_processor.convert_to_mel_spec(waveform)
        return (text, mel)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        wave_id = row['wav']
        text_mel = self.cache.get(wave_id)
        if text_mel is None:
            text_mel = self.get_item(row)
            self.cache[wave_id] = text_mel
            
        return text_mel
    
    def __len__(self):
        return len(self.df)
    
    
    
    
    @staticmethod
    def text_mel_collate_fn(batch):
        text_length_max = torch.tensor(
            [text.shape[-1] for text, _ in batch], 
            dtype=torch.int32
        ).max()
        mel_length_max = torch.tensor(
            [mel.shape[-1] for _, mel in batch],
            dtype=torch.int32
        ).max()
    
        text_lengths = []
        mel_lengths = []
        texts_padded = []
        mels_padded = []
        for text, mel in batch:
            text_length = text.shape[-1]      
            text_padded = torch.nn.functional.pad(
                text,
                pad=[0, text_length_max-text_length],
                value=0
            )
            mel_length = mel.shape[-1]
            mel_padded = torch.nn.functional.pad(
                mel,
                pad=[0, mel_length_max-mel_length],
                value=0
            )
            text_lengths.append(text_length)    
            mel_lengths.append(mel_length)    
            texts_padded.append(text_padded)    
            mels_padded.append(mel_padded)
        text_lengths = torch.tensor(text_lengths, dtype=torch.int32)
        mel_lengths = torch.tensor(mel_lengths, dtype=torch.int32)
        texts_padded = torch.stack(texts_padded, 0)
        mels_padded = torch.stack(mels_padded, 0).transpose(1, 2)
        stop_token_padded = mask_from_seq_lengths(
            mel_lengths,
            mel_length_max
        )
        stop_token_padded = (~stop_token_padded).float()
        stop_token_padded[:, -1] = 1.0
    
        return texts_padded, \
            text_lengths, \
            mels_padded, \
            mel_lengths, \
            stop_token_padded

        
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def load_data(self):
        df = pd.read_csv(self.config.csv_path)
        return df
    
    def split_data(self, data):
        train_df, test_df = train_test_split(data, test_size=0.4)
        return train_df, test_df
    
    def create_dataset(self, train_df, test_df):
        train_dataset = torch.utils.data.DataLoader(
            TextMelDataset(train_df, self.config),
            num_workers = 2,
            shuffle = True,
            sampler = None,
            batch_size = 1,
            pin_memory = True,
            drop_last = True,
            collate_fn = TextMelDataset.text_mel_collate_fn
        )
        
        test_dataset = torch.utils.data.DataLoader(
            TextMelDataset(test_df, self.config),
            num_workers = 2,
            shuffle = True,
            sampler = None,
            batch_size = 1,
            pin_memory = True,
            drop_last = True,
            collate_fn = TextMelDataset.text_mel_collate_fn
        )
        
        return  train_dataset, test_dataset
        
    
    def save_datasets(self, train_dataset, test_dataset):
        os.makedirs(self.config.root_dir, exist_ok=True)

        train_dataset_path = os.path.join(self.config.root_dir, 'train_dataset.pt')
        test_dataset_path = os.path.join(self.config.root_dir, 'test_dataset.pt')

        try:
            torch.save(train_dataset, train_dataset_path)
            torch.save(test_dataset, test_dataset_path)

            logger.info(f"Train dataset saved at: {train_dataset_path}")
            logger.info(f"Test dataset saved at: {test_dataset_path}")
        except Exception as e:
            logger.error(f"Error saving datasets: {str(e)}")
            raise e
    