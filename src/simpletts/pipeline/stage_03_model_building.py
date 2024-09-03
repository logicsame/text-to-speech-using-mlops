from src.simpletts.components.model_building import BuildModel
from src.simpletts.config.configuration import ConfigurationManager

class ModelBuildingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        model_config = config_manager.get_model_building_config()
        model_builder = BuildModel(config=model_config)
        model = model_builder.build_and_save()