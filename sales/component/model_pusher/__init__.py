import os,sys
from sales.logger import logging
from sales.exception import SalesException
from sales.entity.config_entity import ModelPusherConfig
from sales.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
import shutil

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
                      model_evaluation_artifact:ModelEvaluationArtifact)-> None:
        try:
            logging.info(f"{'<<'*20} Model pusher log started{'>>'*20}")
            self.model_pusher_config=model_pusher_config
            self.model_evaluation_artifact=model_evaluation_artifact
            print("model_pusher_config:",model_pusher_config)
        except Exception as e:
            raise SalesException(e,sys) from e              


    def export_model(self)->ModelPusherArtifact:
        try:
            evaluated_model_file_path=self.model_evaluation_artifact.evaluated_model_path
            export_dir=self.model_pusher_config.export_dir_path
            file_name=os.path.basename(evaluated_model_file_path)

            export_dir_file_path=os.path.join(export_dir,file_name)
            os.makedirs(export_dir,exist_ok=True)
            shutil.copy(src=evaluated_model_file_path,dst=export_dir_file_path)

            model_pusher_artifact=ModelPusherArtifact(is_model_pusher=True, export_model_file_path=export_dir_file_path)
            
            return model_pusher_artifact

        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise SalesException(e,sys) from e        