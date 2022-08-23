from flask import Flask,request,jsonify,render_template,url_for
import pickle
import pandas as pd
import numpy as np
import dill
from sales.util import read_yaml_file
import os

CONFIG_DIR='config'
SCHEMA_FILE_NAME='schema.yaml'


app=Flask(__name__)

with open('model.pkl','rb') as obj_file:
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
    schema=read_yaml_file(file_path=os.path.join(CONFIG_DIR,SCHEMA_FILE_NAME))
    schema_data=schema['columns']
    data=pd.DataFrame(dict(zip(k1,v1)),index=[0])
    for col in list(schema_data.keys()):
        data[col]=data[col].astype(schema_data[col])
    output=model.predict(data)[0]
    print('output:',output)
    return render_template('home.html', prediction_text="The Sales is  {}".format(output))

if __name__=='__main__':
    app.run(debug=True)