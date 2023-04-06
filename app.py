from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import pickle

application = Flask(__name__)
app = application
app.config['TEMPLATES_AUTO_RELOAD']=True

# Importing the model
scaler_model=pickle.load(open("model/scaler.pkl","rb"))
rigid_model=pickle.load(open("model/rigid.pkl","rb"))

@app.route("/")
def index():
    return render_template('index.html',title='Form',FMI="")

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temprature=request.form['Temprature']
        RH=request.form['RH']
        Ws=request.form['Ws']
        Rain=request.form['Rain']
        FFMC=request.form['FFMC']
        DMC=request.form['DMC']
        ISI=request.form['ISI']
        Classes=request.form['Classes']
        Region=request.form['Region']

        scaled_data=scaler_model.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        output=rigid_model.predict(scaled_data)

        FMI="The Value of FMI will be "+str(output[0])
        return render_template("index.html",title='Form',FMI=FMI)
    else:
        return render_template("index.html",title='Form',FMI="")
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
