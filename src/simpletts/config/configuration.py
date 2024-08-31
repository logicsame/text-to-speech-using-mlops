from src.simpletts.constants import *
from src.simpletts.utils.common import create_directories, read_yaml
from src.simpletts.entity.config_entity import DataIngestionConfig
from src.simpletts.entity.config_entity import DataTransformationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            wave_path = config.wave_path,
            csv_path = config.csv_path,
            sr  = self.params.sr,
            n_fft = self.params.n_fft,
            n_stft = self.params.n_stft,
            frame_length = self.params.frame_length,
            win_length = self.params.win_length,
            mel_freq = self.params.mel_freq,
            max_mel_time = self.params.max_mel_time,
            max_db = self.params.max_db,
            scale_db = self.params.scale_db,
            ref = self.params.ref,
            power = self.params.power,
            norm_db = self.params.norm_db,
            ampl_multiplier = self.params.ampl_multiplier,
            ampl_amin = self.params.ampl_amin,
            db_multiplier = self.params.db_multiplier,
            ampl_ref = self.params.ampl_ref,
            ampl_power = self.params.ampl_power,
            min_level_db = self.params.min_level_db,
            frame_shift=self.params.frame_shift,
            hop_length=self.params.hop_length
        )
        
        
        return data_transformation_config
    
        