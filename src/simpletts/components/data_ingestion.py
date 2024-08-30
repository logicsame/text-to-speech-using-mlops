import os
import zipfile
from src.simpletts.logging import logger
from tqdm.notebook import tqdm
from dataclasses import replace
from src.simpletts.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def load_file(self):
        if os.path.exists(self.config.source_dir):
            self.config = replace(self.config, local_data_file=self.config.source_dir)
            logger.info(f'File Found at: {self.config.local_data_file}')
        else:
            logger.info(f"File not found at {self.config.source_dir}")
            raise FileNotFoundError(f'No file found at {self.config.source_dir}')
        
        
        
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # open the zip file
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            
            for file in tqdm(iterable=zip_ref.infolist(), total=total_files, desc='Extracting files'):
                zip_ref.extract(member=file, path=unzip_path)
                
            logger.info(f'Extacted {self.config.local_data_file} to {unzip_path}')
            
            