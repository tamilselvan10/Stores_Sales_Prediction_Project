import os,sys
from sales.component import data_ingestion
from sales.component import data_transformation
from sales.entity.config_entity import ModelEvaluationConfig
from sales.exception import SalesException
from sales.logger import logging
from sales.config import Configuration
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact, ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
from sales.component.data_ingestion import DataIngestion
from sales.component.data_validation import DataValidation
from sales.component.data_transformation import DataTransformation
from sales.component.model_trainer import ModelTrainer
from sales.component.model_evalutaion import ModelEvaluation
from sales.component.model_pusher import ModelPusher


class Pipeline:

    def __init__(self,config:Configuration):
        try:
            self.config=config
        except Exception as e:
            raise SalesException(e,sys) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion_config=self.config.get_data_ingestion_config()
            data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise SalesException(e,sys) from e   

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation_config=self.config.get_data_validation_config()
            data_ingestion_artifact=data_ingestion_artifact
            data_validation=DataValidation(data_validation_config=data_validation_config,
                                           data_ingestion_artifact=data_ingestion_artifact)

            return data_validation.initiate_data_validation()                               
        except Exception as e:
            raise SalesException(e,sys) from e   

    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,
                                       data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config=self.config.get_data_transformation_config()
            data_transformation=DataTransformation(data_transformation_config=data_transformation_config,
                                                   data_ingestion_artifact=data_ingestion_artifact,
                                                   data_validation_artifact=data_validation_artifact)
            return data_transformation.initiate_data_transformation()                                       
        except Exception as e:
            raise SalesException(e,sys) from e

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config=self.config.get_model_trainer_config()
            data_transformation_artifact=data_transformation_artifact
            model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,
                                       data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer()                           
        except Exception as e:
            raise SalesException(e,sys) from e     

    def start_model_evaluation(self
                                   ,data_ingestion_artifact:DataIngestionArtifact
                                   ,data_validation_artifact:DataValidationArtifact
                                   ,model_trainer_artifact:ModelTrainerArtifact)->ModelEvaluationArtifact:
        try:
            model_evaluation=ModelEvaluation(model_evaluation_config=self.config.get_model_evaluation_config(),
                            data_ingestion_artifact=data_ingestion_artifact,
                            data_validation_artifact=data_validation_artifact,
                            model_trainer_artifact=model_trainer_artifact)

            return model_evaluation.initiate_model_evaluation()                

        except Exception as e:
            raise SalesException(e,sys) from e        

    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher=ModelPusher(model_pusher_config=self.config.get_model_pusher_config(),
                        model_evaluation_artifact=model_evaluation_artifact)
            return model_pusher.initiate_model_pusher()            
        except Exception as e:
            raise SalesException(e,sys) from e                                     

    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            print(f'\n data_ingestion_artifact:{data_ingestion_artifact}')
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            print(f'\n data_validation_artifact:{data_validation_artifact}')
            data_transformation_artifact=self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact)
            print(f'\n data_transformation_artifact:{data_transformation_artifact}')
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            print(f'\n model_trainer_artifact:{model_trainer_artifact}')
            model_evaluation_artifact=self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                  data_validation_artifact=data_validation_artifact,
                                                                  model_trainer_artifact=model_trainer_artifact)
           
            print(f'\n model_evaluation_artifact:{model_evaluation_artifact}')
            model_pusher_artifact=self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            print(f'model_pusher_artifact:{model_pusher_artifact}')
        except Exception as e:
            raise SalesException(e,sys) from e       

