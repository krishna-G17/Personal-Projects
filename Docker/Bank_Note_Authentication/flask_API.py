# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:33:04 2021

@author: golla
"""
from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__) # to determine from which step the flask has to begin
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

# function to display the web page for the attained IP addresss

@app.route('/')
def welcome():
    return "Welcome ALL"

@app.route('/predict')    
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "the predicted value is "+str(prediction)
    
    
@app.route('/predict_for_file', methods = ["POST"])    
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction  = classifier.predict(df_test)
    
    return "the predicted values for given csv file are "+str(prediction)
        

    
if __name__ == '__main__':
    app.run()
