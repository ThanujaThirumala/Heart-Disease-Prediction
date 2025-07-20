import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
data=pd.read_csv("framingham.csv")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Select only 5 features
x = data_cleaned[['age', 'totChol', 'glucose', 'sysBP', 'heartRate']]
y = data_cleaned['TenYearCHD']

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(x_scaled, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)