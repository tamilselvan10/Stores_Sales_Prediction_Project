from collections import namedtuple

DataIngestionArtifact=namedtuple("DataIngestionArtifact",
["is_ingested","message","train_file_path","test_file_path"])

DataValidationArtifact=namedtuple("DataValidationArtifact",
["is_validated","message","schema_file_path","report_file_path","report_page_file_path"])

DataTransformationArtifact=namedtuple("DataTransformationArtifact",
["is_transformed","message","transformed_train_file_path","transformed_test_file_path","preprocessed_object_file_path"])

ModelTrainerArtifact=namedtuple("ModelTrainerArtifact",
["is_trained","message","trained_model_file_path","train_rmse","test_rmse","train_accuracy","test_accuracy","model_accuracy"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path"])

ModelEvaluationArtifact=namedtuple("ModelEvaluationArtifact",
                                 ["is_model_accepted","evaluated_model_path"])