import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash,send_from_directory
import tensorflow as tf
import base64
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ast
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
from database import *
from pathlib import Path
import pandas as pd
import joblib
import pickle

# Load the trained SVM model
import subprocess
 

app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/uploads/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/")
def home():
    return render_template("main.html")
@app.route("/bhome")
def bhome():
    return render_template("bhome.html")
@app.route("/bl")
def bl():
    return render_template("blogin.html")
@app.route("/uhome")
def uhome():
    return render_template("uhome.html")
@app.route("/ul")
def ul():
    return render_template("ulogin.html")
@app.route("/ur")
def ur():
    return render_template("ureg.html")
@app.route("/br")
def br():
    return render_template("breg.html")
@app.route("/log")
def ll():
    return render_template("main.html")
@app.route("/p")
def p():
    return render_template("p.html")
@app.route("/cp")
def cp():
    return redirect(url_for('details'))
@app.route("/vp")
def vp():
    return render_template("upload.html")
@app.route("/cvp")
def cvp():
    return render_template("check.html")
@app.route("/bregister",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = Buyer_reg(username,email,password) 
        if status == 1:
            return render_template("blogin.html")
        else:
            return render_template("breg.html",m1="failed")        
    

@app.route("/blogin",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = Buyer_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1: 
            session['username'] = request.form['username']                                     
            return render_template("bhome.html", m1="sucess")
        else:
            return render_template("blogin.html", m1="Login Failed")




@app.route("/uregister",methods=['POST','GET'])
def usignup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password) 
        if status == 1:
            return render_template("ulogin.html")
        else:
            return render_template("ureg.html",m1="failed")        
    

@app.route("/ulogin",methods=['POST','GET'])
def ulogin():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1: 
            session['username'] = request.form['username']                                     
            return render_template("uhome.html", m1="sucess")
        else:
            return render_template("ulogin.html", m1="Login Failed")







@app.route("/pre", methods=['POST', 'GET'])
def pre():
    try:
        # Ensure session key exists
        if 'username' not in session:
            return render_template("error.html", error="User session expired. Please log in again.")

        username = session['username']
        features = request.form.get('inputData', '')

        if not features:
            return render_template("error.html", error="No input data provided.")

        # Convert input string to list of floats
        features_list = [float(x) for x in features.split()]

        # Load dataset
        file_path = "dataset 3 kelas.xlsx"  # Ensure the file exists
        df = pd.read_excel(file_path, sheet_name="Sheet1")

        # Handle missing values
        imputer = SimpleImputer(strategy="most_frequent")
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Separate features and target
        if "Academic_Performance" not in df_imputed.columns:
            return render_template("error.html", error="Dataset missing required 'Academic_Performance' column.")

        X = df_imputed.drop(columns=["Academic_Performance"])
        y = df_imputed["Academic_Performance"]

        # Ensure correct number of input features
        if len(features_list) != X.shape[1]:
            return render_template("error.html", error="Invalid number of input features.")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Apply SMOTE to balance dataset
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Train XGBoost model
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
        xgb_model.fit(X_train_resampled, y_train_resampled)

        # Predict on new input
        input_data = np.array([features_list])
        input_df = pd.DataFrame(input_data, columns=X.columns)

        # Make prediction
        prediction = xgb_model.predict(input_df)

        # Interpret prediction
        result = {
            0: "Student Performance is Less",
            1: "Student Performance is Medium",
            2: "Student Performance is High"
        }.get(prediction[0], "Unknown Performance Level")

        # Store prediction results
        poniti(username, features, result)

        return render_template("result.html", text=result)

    except Exception as e:
        print(f"Error: {e}")  # Log error to console
        return render_template("error.html", error=str(e))

















@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("index"))
    username=session['username']
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith((".mp4", ".avi", ".mov")):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{session['username']}_{timestamp}.mp4"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            # Run analyze_fatigue in a subprocess
            result_process = subprocess.run(
                ["python", "process_video.py", filepath], capture_output=True, text=True
            )
            filepath2=filepath = os.path.join('static/uploads/outputs', filename)
            # Capture the result from subprocess
            print(result_process)
            results = result_process.stdout.strip()
            ad(username,filename,results)
            results=ast.literal_eval(results)
            print(results)
            print(result_process)
            return render_template("result1.html", results=results, video_path=filepath2)



@app.route("/details")
def details():
    if "username" not in session:
        return redirect(url_for("index"))  # Redirect to login if no session

    username = session["username"]
    user_data,rf = get_user_data(username)

    if not user_data:
        return render_template("details.html", error="User not found.", username=username)

    return render_template("details.html", user_data=user_data, username=username,rf=rf)


@app.route("/details2", methods=["GET", "POST"])
def details2():
    username = request.form.get("username") or request.args.get("username")

    if not username:
        return redirect(url_for("index"))  # Redirect if no username is provided

    user_data,rf = get_user_data(username)

    if not user_data:
        return render_template("d2.html", error="User not found.", username=username)

    return render_template("d2.html", user_data=user_data, username=username,rf=rf)

if __name__ == "__main__":
    app.run(debug=True)