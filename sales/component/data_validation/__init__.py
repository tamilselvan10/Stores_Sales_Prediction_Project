import os,sys
from sales.constant import DOMAIN_VALUE, NUMERICAL_COLUMNS, ORDINAL_CATEGORICAL_COLUMNS
from sales.exception import SalesException
from sales.logger import logging
from sales.entity.config_entity import DataValidationConfig
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import pandas as pd
from sales.util import read_yaml_file
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json

class DataValidation:

    def __init__(self,data_validation_config:DataValidationConfig,
                      data_ingestion_artifact:DataIngestionArtifact)->None:
        try:
            logging.info(f"{'<<'*20}Data validation log started{'>>'*20}")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            print(f'\nData Validation Config:{self.data_validation_config}')
        except Exception as e:
            raise SalesException(e,sys) from e  

    def is_train_test_file_exists(self)->bool:
        try:
            train_file_exist=False
            test_file_exist=False

            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            train_file_exist=os.path.exists(train_file_path)
            test_file_exist=os.path.exists(test_file_path)

            is_available=train_file_exist and test_file_exist

            if not is_available:
                logging.info(f'Train{[train_file_path]} or Test {[test_file_path]} File is not present')

            return is_available    

        except Exception as e:
            raise SalesException(e,sys) from e 

    def get_train_test_data(self)->pd.DataFrame:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            
            train_data=pd.read_csv(train_file_path)
            test_data=pd.read_csv(test_file_path)

            return train_data,test_data

        except Exception as e:
            raise SalesException(e,sys) from e        

    def check_no_of_columns(self)->bool:
        try:
            logging.info('---No.of.Columns Check---')
            train_data,test_data=self.get_train_test_data()

            train_columns=train_data.shape[1]
            test_columns=test_data.shape[1]    

            logging.info(f'Train columns:{train_columns}')
            logging.info(f'Test columns:{test_columns}')       

            if train_columns==test_columns:
                logging.info('Result: Match')  
                return True                
            else:
                logging.info('Result: Not Match')   
                return False    

        except Exception as e:
            raise SalesException(e,sys) from e               

    def check_column_names(self)->bool:
        try:

            logging.info('---Column names check---')
            train_data,test_data=self.get_train_test_data()

            train_columns=train_data.columns.to_list()
            test_columns=test_data.columns.to_list()

            logging.info(f'Train columns:{train_columns}')
            logging.info(f'Test columns:{test_columns}')

            if train_columns==test_columns:
                logging.info('Result: Match') 
                return True
            else:
                logging.info('Result: Not Match') 
                return False    

        except Exception as e:
            raise SalesException(e,sys) from e

    def check_missing_values(self)->bool:
        try:
            logging.info('---Missing Values Check:')

            train_data,test_data=self.get_train_test_data()
            
            logging.info('---Train Dataset:')
            train_result=True
            for col in train_data.columns.to_list():
                value=None
                pct=None
                value=train_data[col].isnull().sum()
                pct=round(train_data[col].isnull().sum()/train_data.shape[0]*100,2)

                if value>0:
                    train_result=False
                
                logging.info(f'col:{col},value:{value},pct:{pct}')  

            logging.info('---Test Dataset:')
            test_result=True
            for col in test_data.columns.to_list():
                value=None
                pct=None
                value=test_data[col].isnull().sum()
                pct=round(test_data[col].isnull().sum()/test_data.shape[0]*100,2)  

                if value>0:
                    test_result=False

                logging.info(f"col:{col},value:{value},pct:{pct}")       

            return train_result and test_result

        except Exception as e:
            raise SalesException(e,sys) from e  

    def check_outliers(self)->bool:
        try:

            logging.info('---Outliers Check---')
            train_data,test_data=self.get_train_test_data()
            scheme_data=read_yaml_file(file_path=self.data_validation_config.schema_file_path)
            numerical_columns=scheme_data[NUMERICAL_COLUMNS]
            lower=-0.5
            upper=0.5

            logging.info('---Train Dataset:')            
            train_result=True
            for col in numerical_columns:
                skew=None
                result=None
                skew=round(train_data[col].skew(),2)

                if skew<lower or skew>upper:
                    result='Outliers'
                    train_result=False
                else:
                    result='No Outliers'  

                logging.info(f'col:{col},skew:{skew},result:{result}')


            logging.info('--- Test Dataset:')
            test_result=True
            for col in numerical_columns:
                skew=None
                result=None
                skew=round(test_data[col].skew(),2)

                if skew<lower or skew>upper:
                    result='Outliers'
                    test_result=False
                else:
                    result='No Outliers'

                logging.info(f'col:{col},skew:{skew},result:{result}')       

            return train_result and test_result             

        except Exception as e:
            raise SalesException(e,sys) from e 

    def check_domain_value(self)->bool:
        try:
            logging.info('---Domain Value Check---')

            train_data,test_data=self.get_train_test_data()
            schema_data=read_yaml_file(file_path=self.data_validation_config.schema_file_path)
            domain_value=schema_data[DOMAIN_VALUE]
            
            logging.info('---Train Dataset:')
            train_result=True
            for col in schema_data[ORDINAL_CATEGORICAL_COLUMNS]:
                logging.info(f'col:{col}')
                logging.info(f'Train:{sorted(list(train_data[col].dropna().unique()))}')
                logging.info(f'Domain:{sorted(domain_value.get(col))}')
                if sorted(list(train_data[col].dropna().unique()))==sorted(domain_value.get(col)):
                    logging.info('Result: Match')
                else:
                    logging.info('Result: Not Match')
                    train_result=False  

            logging.info('---Test Dataset:')
            test_result=True
            for col in schema_data[ORDINAL_CATEGORICAL_COLUMNS]:
                logging.info(f'col:{col}')
                logging.info(f'Test:{sorted(list(test_data[col].dropna().unique()))}')
                logging.info(f'Domain:{sorted(domain_value.get(col))}')

                if sorted(list(test_data[col].dropna().unique()))==sorted(domain_value.get(col)):
                    logging.info('Result: Match')
                else:
                    logging.info('Result: Not Match')
                    test_result=False

            return train_result and test_result        

        except Exception as e:
            raise SalesException(e,sys) from e  

    def json_report(self):
        try:

            profile=Profile(sections=[DataDriftProfileSection()])
            train_data,test_data=self.get_train_test_data()
            profile.calculate(train_data,test_data)

            report=json.loads(profile.json())

            report_file_path=self.data_validation_config.report_file_path

            report_file_dir=os.path.dirname(report_file_path)

            os.makedirs(report_file_dir,exist_ok=True)

            with open(report_file_path,'w') as report_file:
                json.dump(report,report_file,indent=6)                

        except Exception as e:
            raise SalesException(e,sys) from e

    def html_report(self):
        try:
            dashboard=Dashboard(tabs=[DataDriftTab()])

            train_data,test_data=self.get_train_test_data()

            dashboard.calculate(train_data,test_data)

            report_page_file_path=self.data_validation_config.report_page_file_path
            report_page_file_dir=os.path.dirname(report_page_file_path)

            os.makedirs(report_page_file_dir,exist_ok=True)

            dashboard.save(report_page_file_path)

        except Exception as e:
            raise SalesException(e,sys) from e
    def check_data_drift(self):
        try:
            self.json_report()
            self.html_report()
        except Exception as e:
            raise SalesException(e,sys) from e                          

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            self.is_train_test_file_exists()
            self.check_no_of_columns()
            self.check_column_names()
            self.check_missing_values()
            self.check_outliers()
            self.check_domain_value()
            self.check_data_drift()

            data_validation_artifact=DataValidationArtifact(is_validated=True, 
                                   message='Data Validation Completed Successfully', 
                                   schema_file_path=self.data_validation_config.schema_file_path,
                                   report_file_path=self.data_validation_config.report_file_path, 
                                   report_page_file_path=self.data_validation_config.report_page_file_path)

            logging.info(f'data_validation_artifact:{data_validation_artifact}')
            return data_validation_artifact                       
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info(f"{'<<'*20}Data Validation log completed{'>>'*20}")
