from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__)
app = application

#importing models with pickle 
ridge_model = pickle.load(open('models/ridge.pickle','rb'))
standard_scaler = pickle.load(open('models/scaler.pickle','rb'))

@app.route("/")
def Hello_world():
    return render_template("/index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        Ws = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))
        
        new_data_scaled = standard_scaler.transform([[temperature,rh,Ws,Rain,ffmc,dmc,isi,Classes,region]])
        res = ridge_model.predict(new_data_scaled)

        return render_template("/home.html",result = res[0])

    else:
        return render_template("/home.html")

if __name__ =="__main__":
    
    app.run(host="0.0.0.0",debug =True)