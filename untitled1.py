# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:08:51 2021

@author: shubh
"""

from flask import Flask,render_template,request

app = Flask(__name__,template_folder='C:\\Users\\shubh\\CFDDNN\\templates')

import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from tensorflow.keras.models import model_from_json
#from sklearn.preprocessing import StandardScaler
import pickle
sc = pickle.load(open('scaler.pkl','rb'))
#app.debug = True

@app.route('/')
def hello():
    return render_template('home.html')
    
@app.route('/input',methods=['GET','POST'])
def input():
    pred11=[[]]
    if request.method=='POST':
        cc_num=request.form.get("cc_num")
        Category=request.form.get("Category")
        Amount=request.form.get("Amount")
        Gender=request.form.get("Gender")
        Street=request.form.get("Street")
        State=request.form.get("State")
        PIN=request.form.get("PIN")
        lat=request.form.get("lat")
        long=request.form.get("long")
        CityPOP=request.form.get("CityPOP")
        Job=request.form.get("Job")
        unix=request.form.get("unix")
        merchlat=request.form.get("merchlat")
        merchlong=request.form.get("merchlong")
    
        data=[[cc_num,Category,Amount, Gender,Street, State, PIN, lat, long, CityPOP, Job,unix,merchlat,merchlong]]
        data = np.array(data)
        data[:,1] = le.fit_transform(data[:,1])
        print(data)
        data[:,3] = le.fit_transform(data[:,3])
        print(data)
        
        data[:,4] = le.fit_transform(data[:,4])
        print(data)
        
        data[:,5] = le.fit_transform(data[:,5])
        print(data)
        
        data[:,10] = le.fit_transform(data[:,10])
        print(data)
        
        



        data = sc.transform(data)
        print(data)
        json_file = open('final_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("final_model.h5")
        print("Loaded model from disk")


        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #score = loaded_model.evaluate(X_test, y_test, verbose=0)
        pred11 = loaded_model.predict(data)
        print(pred11)
    
    
    
    return render_template('input.html',results=pred11)
    
    
    
if __name__ == '__main__':
    app.run(debug=True)