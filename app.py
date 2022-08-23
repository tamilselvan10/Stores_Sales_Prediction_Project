from flask import Flask,request,jsonify,render_template,url_for
import pickle
import pandas as pd
import numpy as np
import dill
from sales.util import read_yaml_file
import os
from sales.constant import *

MODEL_DIR='saved_models'
MODEL_FILE_LIST=[eval(i) for i in os.listdir(os.path.join(MODEL_DIR))]
MODEL_FILE_DIR=os.path.join(MODEL_DIR,str(max(MODEL_FILE_LIST)))
MODEL_FILE_NAME=os.listdir(MODEL_FILE_DIR)[0]

MODEL_FILE_PATH=os.path.join(MODEL_FILE_DIR,MODEL_FILE_NAME)

print(F'MODEL_FILE_PATH:{MODEL_FILE_PATH}')

app=Flask(__name__)

with open(MODEL_FILE_PATH,'rb') as obj_file:
    model=dill.load(obj_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print('data')
    new_data=pd.DataFrame(dict(data),index=[0])
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    k1=list(request.form.keys())
    v1=list(request.form.values())
    CONFIG_DIR='config'
    SCHEMA_FILE_NAME='schema.yaml'
    schema=read_yaml_file(file_path=os.path.join(CONFIG_DIR,SCHEMA_FILE_NAME))
    schema_data=schema[COLUMNS]
    target_columns=schema[TARGET_COLUMNS]
    del schema_data[target_columns]
    data=pd.DataFrame(dict(zip(k1,v1)),index=[0])
    for col in schema_data.keys():
        data[col]=data[col].astype(schema_data[col])
    output=model.predict(data)[0]
    print('output:',output)
    return render_template('home.html', prediction_text="The Sales is  {}".format(output))

if __name__=='__main__':
    app.run(debug=True)