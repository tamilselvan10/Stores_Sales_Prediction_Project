import os,sys
from tkinter.filedialog import test
from sales.exception import SalesException
from sales.logger import logging
from sales.entity.config_entity import DataIngestionConfig
from sales.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'<<'*20} Data ingestion log started {'>>'*20}")
            self.data_ingestion_config=data_ingestion_config
            print('Data Ingestion config:',self.data_ingestion_config)
        except Exception as e:
            raise SalesException(e,sys) from e

    def download_data(self):
        try:
            file_path=r'E:\ML\stores\artifact\data_ingestion\20-08-2022\raw_data\data.csv'
            data=pd.read_csv(file_path)
            raw_data_dir=self.data_ingestion_config.raw_data_dir
            file_name=os.path.basename(file_path)
            raw_file_path=os.path.join(raw_data_dir,file_name)
            os.makedirs(raw_data_dir,exist_ok=True)
            data.to_csv(raw_file_path,index=False)
        except Exception as e:
            raise SalesException(e,sys) from e        

    def split_data_as_train_test(self)->DataIngestionArtifact:
        try:
            raw_data_dir=self.data_ingestion_config.raw_data_dir
            file_name=os.listdir(raw_data_dir)[0]
            file_path=os.path.join(raw_data_dir,file_name)

            sales=pd.read_csv(file_path)

            sales['sales_cat']=pd.cut(sales['Item_Outlet_Sales'],
                                   bins=[0,1000,5000,10000,np.inf],
                                   labels=[1,2,3,4])

            logging.info('Splitting data into train and test')                       
            
            strat_train_set=None
            strat_test_set=None

            split=StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)

            for train_index,test_index in split.split(sales,sales['sales_cat']):
                strat_train_set=sales.loc[train_index].drop(columns='sales_cat',axis=1)
                strat_test_set=sales.loc[test_index].drop(columns='sales_cat',axis=1)

           
            train_file_path=os.path.join(self.data_ingestion_config.ingested_train_dir,file_name)
            test_file_path=os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)

            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f'Exporting Train dataset into file:{train_file_path}')
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
                logging.info(f'Exporting Test dataset into file:{test_file_path}')
                strat_test_set.to_csv(test_file_path,index=False)

            data_ingestion_artifact=DataIngestionArtifact(is_ingested=True,
                                  message='Data Ingestion Completed Successfully',
                                  train_file_path=train_file_path,
                                  test_file_path=test_file_path)     

            logging.info(f'Data Ingestion Artifact:{data_ingestion_artifact}')
            return data_ingestion_artifact                         




        except Exception as e:
            raise SalesException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.download_data()
            return self.split_data_as_train_test()
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info(f"{'<<'*20}Data ingestion log completed{'>>'*20}") 