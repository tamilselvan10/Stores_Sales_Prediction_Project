from importlib.resources import read_binary
from lib2to3.pytree import Base
from msilib import schema
import os,sys
from sales.logger import logging
from sales.exception import SalesException
from sales.entity.config_entity import DataTransformationConfig
from sales.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
from sales.constant import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sales.util import *
from sales.constant import *
import numpy as np


class OutlierRemover(BaseEstimator,TransformerMixin):
    
    def __init__(self,factor=1.5):
        self.factor=factor

    def outlier_detector(self,X,y=None):
        try:
            X=pd.Series(X).copy()
            q1=X.quantile(0.25)
            q3=X.quantile(0.75)
            iqr=q3-q1
            self.lower_bound.append(q1-self.factor*iqr)
            self.upper_bound.append(q3+self.factor*iqr)

        except Exception as e:
            raise SalesException(e,sys) from e  

    def fit(self,X,y=None):
        try:
            self.lower_bound=list()
            self.upper_bound=list()
            X.apply(self.outlier_detector)
            return self
        except Exception as e:
            raise SalesException(e,sys) from e  

    def transform(self,X,y=None):
        try:
            X=pd.DataFrame(X).copy()
            for index in range(X.shape[1]):
                x=X.iloc[:,index].copy()
                x[x<self.lower_bound[index]]=self.lower_bound[index]
                x[x>self.upper_bound[index]]=self.upper_bound[index]
                X.iloc[:,index]=x
            return X    
        except Exception as e:
            raise SalesException(e,sys) from e    


class feature_generator(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        try:
            self.columns = columns
        except Exception as e:
            raise SalesException(e, sys) from e

    def fit(self, X, y=None):
        try:
            return self
        except Exception as e:
            raise SalesException(e, sys) from e

    def transform(self, X, y=None):
        try:
            x = X.copy()
            x[OUTLET_AGE_COLUMN] = YEAR-x[self.columns]
            x.drop(columns=self.columns, axis=1, inplace=True)
            return x
        except Exception as e:
            raise SalesException(e, sys) from e

class domain_value(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def fit(self,X,y=None):
        try:
            return self
        except Exception as e:
            raise SalesException(e,sys) from e

    def transform(self,X,y=None):
        try:
            x=X.copy()
            x['Item_Fat_Content'].replace({'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'},inplace=True)
            return x
        except Exception as e:
            raise SalesException(e,sys) from e        


class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig,
                      data_ingestion_artifact:DataIngestionArtifact,
                      data_validation_artifact:DataValidationArtifact)->None:

        try:
            logging.info(f"{'<<'*20} Data Transformation log started {'>>'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            print(f'data_transformation_config:{data_transformation_config}')
        except Exception as e:
            raise SalesException(e,sys) from e

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            
            schema_data=read_yaml_file(file_path=self.data_validation_artifact.schema_file_path)
            numerical_columns=schema_data[NUMERICAL_COLUMNS]
            nom_cat_columns=schema_data[NOMINAL_CATEGORICAL_COLUMNS]
            ord_cat_columns=schema_data[ORDINAL_CATEGORICAL_COLUMNS]
            year_columns=[schema_data[YEAR_COLUMNS]]

            num_pipeline=Pipeline(steps=[
                                  ('OutlierRemover',OutlierRemover()),
                                  ('imputer',SimpleImputer(strategy='median')),
                                  ('scaler', StandardScaler())])

            nom_cat_pipeline=Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='most_frequent')),
                                      ('one_hot_encoder', OneHotEncoder()),
                                      ('scaler', StandardScaler(with_mean=False))
                                      ])
            ord_cat_pipeline=Pipeline(steps=[
                                    ('domain_value',domain_value()),
                                    ('imputer', SimpleImputer(strategy='most_frequent')),
                                    ('LabelEncoder',OrdinalEncoder()),
                                    ('scaler', StandardScaler(with_mean=False))
                                        ])           

            year_pipeline=Pipeline(steps=[
                                   ('featuregenerator',feature_generator(columns=year_columns))
                                  ])                                           

            preprocessing=ColumnTransformer([
                                           ('num_pipeline',num_pipeline,numerical_columns),
                                           ('nom_cat_pipeline',nom_cat_pipeline,nom_cat_columns),
                                           ('ord_cat_pipeline',ord_cat_pipeline,ord_cat_columns),
                                           ('year_pipeline',year_pipeline,year_columns)
                                          ])

            return preprocessing                              
        except Exception as e:
            raise SalesException(e,sys) from e 



    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            schema_file_path=self.data_validation_artifact.schema_file_path

            train_df=load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_df=load_data(file_path=test_file_path,schema_file_path=schema_file_path)

            schema_data=read_yaml_file(file_path=schema_file_path)
            id_columns=schema_data[ID_COLUMNS]
            target_columns=[schema_data[TARGET_COLUMNS]]
            
            feature_train_df=train_df.drop(columns=id_columns+target_columns,axis=1)
            target_train_df=train_df[target_columns]

            feature_test_df=test_df.drop(columns=id_columns+target_columns,axis=1)
            target_test_df=test_df[target_columns]

            preprocessing_object=self.get_data_transformer_object()

            train_arr=preprocessing_object.fit_transform(feature_train_df)
            test_arr=preprocessing_object.transform(feature_test_df)

            train_arr=np.c_[train_arr,np.array(target_train_df)]
            test_arr=np.c_[test_arr,np.array(target_test_df)]

            preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path=preprocessed_object_file_path,obj=preprocessing_object)

            transformed_train_dir=self.data_transformation_config.transformed_train_dir
            transformed_test_dir=self.data_transformation_config.transformed_test_dir

            train_file_name=os.path.basename(train_file_path).replace('.csv','.npz')
            test_file_name=os.path.basename(test_file_path).replace('.csv','.npz')

            transformed_train_file_path=os.path.join(transformed_train_dir,train_file_name)
            transformed_test_file_path=os.path.join(transformed_test_dir,test_file_name)

            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            data_transformation_artifact=DataTransformationArtifact(is_transformed=True, 
                                       message="Data Transformation Completed Successfully", 
                                       transformed_train_file_path=transformed_train_file_path, 
                                       transformed_test_file_path=transformed_test_file_path, 
                                       preprocessed_object_file_path=preprocessed_object_file_path)

            return data_transformation_artifact                           
            
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info(f"{'<<'*20} Data Transformation log completed {'>>'*20}")



