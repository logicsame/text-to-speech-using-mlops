from src.simpletts.pipeline.stage_01_data_ingestion import DataIngestionpipeline
from src.simpletts.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.simpletts.pipeline.stage_03_model_building import ModelBuildingPipeline
from src.simpletts.logging import logger



STAGE_NAME = 'Data Ingestion Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionpipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
STAGE_NAME = 'Data Transformation Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
STAGE_NAME = 'Model Building Stage'

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_building = ModelBuildingPipeline()
   model_building.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e