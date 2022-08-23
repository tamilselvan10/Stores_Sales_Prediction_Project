import os,sys
from sales.exception import SalesException
from sales.config import Configuration
from sales.pipeline import Pipeline

pipeline=Pipeline(config=Configuration())
pipeline.run_pipeline()
