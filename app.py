"""
   command to start web server
   set FLASK_ENV=development
   set FLASK_APP=app
   flask run
   
"""

import ssl
import time
import json
import plotly
import numpy as np
import pandas as pd
import tensorflow as tf
import db.dbconnect as dbc
import plotly.express as px

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_nav import Nav
from flask_nav.elements import Navbar, View

##certificate and key files
context = ('conf/server.crt', 'conf/server.key')

app = Flask(__name__, static_folder='C:\\PATH\\TO\\YOUR\\FOLDER\\static')

## initializing Navigations
nav = Nav()

topbar = Navbar('',
    View('Home', 'index'),
    View('Dashboard', 'dashboard'),
    View('Fraud Detection', 'fraud_detection'),
    )

nav.register_element('top', topbar)
nav.init_app(app)

## get data
def get_all_data_pd():
    conn = dbc.create_connection('demo_database.db')
    table = 'emscad_all'
    query = 'SELECT * From '+ table
    df = pd.read_sql(query, conn)
    dbc.close_connection(conn)
    return df

## web app functions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/_get_table')
def get_table():
    df = get_all_data_pd()
    return  jsonify(number_elements=df.shape[0]*df.shape[1], raw_table=df_html)

@app.route('/fraud_detection')
def fraud_detection():
    return render_template('fraud_detection.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        data = get_all_data_pd()
        cleaned_data = clean_data(data)
        result_ = predictor_model(cleaned_data)
        result = []

        for row in result_:
            if row == 0:
                result.append('LEGITIMATE')
            else:
                result.append('FRAUD')

        return render_template('result.html', prediction=result)

## helper functions
def clean_data(data):
    data.loc[data['function'].isnull(),'function'] = data['department']
    data = data.drop(columns=['department'])
    data[['salary_range_min', 'salary_range_max']] = data['salary_range'].str.split('-', 1, expand=True)
    data['salary_range_min'] = data['salary_range_min']
    data['salary_range_min'] = data['salary_range_min'].fillna(-1)
    data['salary_range_min'] = data['salary_range_min'].astype(int)
    data['salary_range_max'] = data['salary_range_max']
    data['salary_range_max'] = data['salary_range_max'].fillna(-1)
    data['salary_range_max'] = data['salary_range_max'].astype(int)
    data[['country', 'state', 'city']] = data['location'].str.split(',', 2, expand=True)
    data = data.drop(columns=['salary_range', 'location', 'in_balanced_dataset'])
    data = data.replace(r'^\s+$', np.nan, regex=True)

    for col in data.columns:
        if 'salary' in col:
            continue
        data[col] = data[col].astype('category').cat.codes

    data = data.drop(columns=['fraudulent'])

    return data

def predictor_model(data):
    loaded_model = tf.keras.models.load_model('static/fraud_detection_mlp.h5')
    result = loaded_model.predict(data, use_multiprocessing=True, workers=4)
    return result


## run app
app.run(host='127.0.0.1', 
        port='8082',
        debug=True, 
        ssl_context=context)
