from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(_name_)

# Load data and train model
data = pd.read_csv("framingham.csv")
imputer = SimpleImputer(strategy='mean')
data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

X = data_cleaned.drop(columns=['TenYearCHD'])
y = data_cleaned['TenYearCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_values = [
        float(request.form['age']),
        float(request.form['totChol']),
        float(request.form['glucose']),
        float(request.form['sysBP']),
        float(request.form['heartRate']),
        # Add more if needed
    ]
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    return render_template('index.html', prediction=int(prediction))

if _name_ == '_main_':
    app.run(debug=True)