# NAME : ARJUN SREENIVAS
# ROLL NO: 2022BCS0060

import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

df = pd.read_csv("dataset/winequality-white.csv",sep=";")

X = df.drop(['quality'],axis=1)
y = df['quality']

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y=y_train)
prediction = model.predict(X_test)

r2 = r2_score(prediction,y_test)
mse = mean_squared_error(prediction,y_test)

os.makedirs("output",exist_ok=True)
joblib.dump(model,"output/model.pkl")

res = {
    "r2" : r2,
    "mse" : mse
}
with open("output/metrics.json","w") as f:
    json.dump(res,f)