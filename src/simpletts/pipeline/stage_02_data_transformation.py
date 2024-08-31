from src.simpletts.components.data_transformation import DataTransformation
from src.simpletts.config.configuration import ConfigurationManager
from src.simpletts.components.data_transformation import AudioProcessor

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        dataset = data_transformation.load_data()
        train_df, test_df = data_transformation.split_data(dataset)
        train_dataset, test_dataset = data_transformation.create_dataset(train_df, test_df)
        data_transformation.save_datasets(train_dataset, test_dataset)
        audio_processor = AudioProcessor(config=data_transformation_config)