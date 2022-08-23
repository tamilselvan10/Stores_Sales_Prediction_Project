import os,sys
from sales.exception import SalesException
from sales.constant import *
from sales.entity.config_entity import DataIngestionConfig, DataTransformationConfig,ModelPusherConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig
from sales.util import read_yaml_file
from sales.logger import logging
from datetime import datetime

class Configuration:

    def __init__(self,
                config_file_path:str=CONFIG_FILE_PATH,
                current_time_stamp:str=CURRENT_TIME_STAMP
                )->None:
                try:
                    self.config_info=read_yaml_file(file_path=config_file_path)
                    self.training_pipeline_config=self.get_training_pipeline_config()
                    self.time_stamp=current_time_stamp
                except Exception as e:
                    raise SalesException(e,sys) from e
    
    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            
            artifact_dir=self.training_pipeline_config.artifact_dir

            data_ingestion_info=self.config_info[DATA_INGESTION_CONFIG_KEY]

            data_ingestion_artifact_dir=os.path.join(artifact_dir,
                                                     DATA_INGESTION_ARTIFACT,
                                                     self.time_stamp)
            raw_data_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            ingested_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR_KEY])
            ingested_train_dir=os.path.join(ingested_dir,data_ingestion_info[DATA_INGESTION_INGESTED_TRAIN_DIR_KEY])
            ingested_test_dir=os.path.join(ingested_dir,data_ingestion_info[DATA_INGESTION_INGESTED_TEST_DIR_KEY])                                         
            
            data_ingestion_config=DataIngestionConfig(raw_data_dir=raw_data_dir,
                                ingested_train_dir=ingested_train_dir,
                                ingested_test_dir=ingested_test_dir)

            logging.info(f'Data ingestion config:{data_ingestion_config}')
            return data_ingestion_config                    
        except Exception as e:
            raise SalesException(e,sys) from e


    def get_data_validation_config(self)-> DataValidationConfig:
        try:
            
            artifact_dir=self.training_pipeline_config.artifact_dir

            data_validation_info=self.config_info[DATA_VALIDATION_CONFIG_KEY]

            data_validation_artifact_dir=os.path.join(artifact_dir,
                                                     DATA_VALIDATION_ARTIFACT,
                                                    self.time_stamp)

            schema_file_path=os.path.join(ROOT_DIR,
                                          data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                          data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])    

            report_file_path=os.path.join(data_validation_artifact_dir,data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])

            report_page_file_path=os.path.join(data_validation_artifact_dir,
                                               data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])                            

            
            data_validation_config=DataValidationConfig(schema_file_path=schema_file_path,
                                                        report_file_path=report_file_path,
                                                        report_page_file_path=report_page_file_path)

            logging.info(f"data_validation_config:{data_validation_config}") 

            return data_validation_config
                                                       
        except Exception as e:
            raise SalesException(e,sys) from e      


    def get_data_transformation_config(self)->DataTransformationConfig:
        try:

            artifact_dir=self.training_pipeline_config.artifact_dir

            data_transformation_artifact_dir=os.path.join(artifact_dir,
                                                        DATA_TRANSFORMATION_ARTIFACT,
                                                        self.time_stamp)
            data_validation_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            
            transformed_dir=os.path.join(data_transformation_artifact_dir,data_validation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY])
            transformed_train_dir=os.path.join(transformed_dir,data_validation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY])
            transformed_test_dir=os.path.join(transformed_dir,data_validation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY])
            preprocessed_file_name=data_validation_info[DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY]
            preprocessed_file_dir=data_validation_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]
            preprocessed_object_file_path=os.path.join(data_transformation_artifact_dir,preprocessed_file_dir,preprocessed_file_name)                                            

            data_transformation_config=DataTransformationConfig(transformed_train_dir=transformed_train_dir,
                                     transformed_test_dir=transformed_test_dir, 
                                     preprocessed_object_file_path=preprocessed_object_file_path)
            
            logging.info(f"data_transformation_config:{data_transformation_config}")

            return data_transformation_config                         
        except Exception as e:
            raise SalesException(e,sys) from e      

    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            artifact_dir=self.training_pipeline_config.artifact_dir
            model_trainer_artifact_dir=os.path.join(artifact_dir,
                                                    MODEL_TRAINER_ARTIFACT,
                                                    self.time_stamp)

            model_trainer_info=self.config_info[MODEL_TRAINER_CONFIG_KEY]
            trained_model_dir=model_trainer_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
            trained_model_file_name=model_trainer_info[MODEL_TRAINER_MODEL_FILE_NAME_KEY]
            base_accuracy=model_trainer_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            config_dir=model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY]
            config_file_name=model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]

            trained_model_file_path=os.path.join(model_trainer_artifact_dir,trained_model_dir,trained_model_file_name)
            model_config_file_path=os.path.join(config_dir,config_file_name)

            model_trainer_config=ModelTrainerConfig(trained_model_file_path=trained_model_file_path, 
                               base_accuracy=base_accuracy, 
                               model_config_file_path=model_config_file_path)

            logging.info(f"model_trainer_config:{model_trainer_config}")
            return model_trainer_config                   
        except Exception as e:
            raise SalesException(e,sys) from e          

    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        try:
            
            artifact_dir=self.training_pipeline_config.artifact_dir

            model_evaluation_artifact_dir=os.path.join(artifact_dir,MODEL_EVALUATION_ARTIFACT)

            model_evaluation_info=self.config_info[MODEL_EVALUATION_CONFIG_KEY]

            model_evaluation_file_path=os.path.join(model_evaluation_artifact_dir,model_evaluation_info[MODEL_EVALUATION_FILE_NAME_KEY])

            model_evaluation_config=ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                 timestamp=self.time_stamp)
            logging.info(f'model_evaluation_config:{model_evaluation_config}')   
            return model_evaluation_config                  
            
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:

            model_pusher_info=self.config_info[MODEL_PUSHER_CONFIG_KEY]
            model_pusher_dir_path=os.path.join(ROOT_DIR,
                                               model_pusher_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                               self.time_stamp)
            model_pusher_config=ModelPusherConfig(export_dir_path=model_pusher_dir_path)

            logging.info(f'model_pusher_config:{model_pusher_config}')

            return model_pusher_config

        except Exception as e:
            raise SalesException(e,sys) from e        

    def get_training_pipeline_config(self)->TrainingPipelineConfig:
        try:
            training_pipeline_info=self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            pipeline_name=training_pipeline_info[TRAINING_PIPELINE_NAME_KEY]
            artifact=training_pipeline_info[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            artifact_dir=os.path.join(ROOT_DIR,pipeline_name,artifact)
            training_pipeline_config=TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f'Training Pipeline Config:{training_pipeline_config}')
            return training_pipeline_config
        except Exception as e:
            raise SalesException(e,sys) from e                



