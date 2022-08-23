import os
from datetime import datetime

ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
CONFIG_FILE_NAME='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

CURRENT_TIME_STAMP=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY="training_pipeline_config"
TRAINING_PIPELINE_NAME_KEY="pipeline_name"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY="artifact_dir"

# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY="data_ingestion_config"
DATA_INGESTION_RAW_DATA_DIR_KEY="raw_data_dir"
DATA_INGESTION_INGESTED_DIR_KEY="ingested_dir"
DATA_INGESTION_INGESTED_TRAIN_DIR_KEY="ingested_train_dir"
DATA_INGESTION_INGESTED_TEST_DIR_KEY="ingested_test_dir"
DATA_INGESTION_ARTIFACT='data_ingestion'

# Data Validation related variable
DATA_VALIDATION_CONFIG_KEY="data_validation_config"
DATA_VALIDATION_SCHEMA_DIR_KEY="schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY="schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY="report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY="report_page_file_name"
DATA_VALIDATION_ARTIFACT='data_validation'

DOMAIN_VALUE="domain_value"
ORDINAL_CATEGORICAL_COLUMNS="ordinal_cat_columns"
NOMINAL_CATEGORICAL_COLUMNS="nominal_cat_columns"
NUMERICAL_COLUMNS="numerical_columns"
YEAR_COLUMNS="year_columns"
ID_COLUMNS="id_columns"
TARGET_COLUMNS="target_columns"

# Data Transformation related variable
DATA_TRANSFORMATION_CONFIG_KEY="data_transformation_config"
DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY="transformed_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY="transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY="transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY="preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY="preprocessed_object_file_name"
DATA_TRANSFORMATION_ARTIFACT='data_transformation'

OUTLET_AGE_COLUMN='Outlet_Age'
YEAR=2013

COLUMNS="columns"

# Model Training related variable
MODEL_TRAINER_CONFIG_KEY="model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY="trained_model_dir"
MODEL_TRAINER_MODEL_FILE_NAME_KEY="model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY="base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY="model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY="model_config_file_name"
MODEL_TRAINER_ARTIFACT="model_trainer"

# Model Evaluation related variable
MODEL_EVALUATION_CONFIG_KEY="model_evaluation_config"
MODEL_EVALUATION_FILE_NAME_KEY="model_evaluation_file_name"
MODEL_EVALUATION_ARTIFACT="model_evaluation"

# Model Pusher related variable
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

