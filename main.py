from src.simpletts.pipeline.stage_01_data_ingestion import DataIngestionpipeline
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