import yaml
from sales.exception import SalesException
import os,sys
import pandas as pd
from sales.constant import *
import dill
import numpy as np



def save_object(file_path:str,obj):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,'wb') as obj_file:
            dill.dump(obj,obj_file)
    except Exception as e:
        raise SalesException(e,sys) from e

def load_object(file_path:str):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SalesException(e,sys) from e        

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise SalesException(e,sys) from e        

def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SalesException(e,sys) from e

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary
    """
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SalesException(e,sys) from e        

def write_yaml_file(file_path:str,data:dict=None):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise SalesException(e,sys) from e          

def load_data(file_path:str,schema_file_path:str)->pd.DataFrame:
    try:
        schema_data=read_yaml_file(file_path=schema_file_path)
        schema=schema_data[COLUMNS]

        df=pd.read_csv(file_path)

        error_message=""
        for col in df.columns.to_list():
            if col in list(schema.keys()):
                df[col].astype(schema.get(col))
            else:
                error_message=f"{error_message} \nColumn:[{col}] is not present in the schema"
        if len(error_message)>0:
            raise Exception(error_message)
        return df    

    except Exception as e:
        raise SalesException(e,sys) from e


