from setuptools import setup,find_packages
from typing import List
from sales.exception import SalesException
import os,sys

#Declaring variables for setup functions
PROJECT_NAME="sales-predictor"
VERSION="0.0.1"
AUTHOR="Tamil Selvan"
DESRCIPTION="Store Sales Prediction Machine Learning Project"

REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements_list()->List[str]:
    try:
        with open(REQUIREMENT_FILE_NAME,'r') as requirement_file:
            requirement_list=requirement_file.readlines()
            requirement_list=[requirement_name.replace('\n','') for requirement_name in requirement_list]
            if HYPHEN_E_DOT in requirement_list:
                requirement_list.remove(HYPHEN_E_DOT)
            return requirement_list

    except Exception as e:
        raise SalesException(e,sys) from e

setup(
name=PROJECT_NAME,
version=VERSION,
author=AUTHOR,
description=DESRCIPTION,
packages=find_packages(),
install_requires=get_requirements_list()
)        

