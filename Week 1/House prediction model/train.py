import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


X = np.load(r"D:\AI and ML lrn\Week 1\House prediction model\data\X.npy")
Y = np.load(r"D:\AI and ML lrn\Week 1\House prediction model\data\y.npy")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

np.save(r'D:\AI and ML lrn\Week 1\House prediction model\data/X_train.npy', X_train)
np.save(r'D:\AI and ML lrn\Week 1\House prediction model\data/Y_train.npy', Y_train)
np.save(r'D:\AI and ML lrn\Week 1\House prediction model\data/X_test.npy', X_test)
np.save(r'D:\AI and ML lrn\Week 1\House prediction model\data/Y_test.npy', Y_test)

Model_fit = LinearRegression()
Model_fit.fit(X_train, Y_train)

Score = Model_fit.score(X_test, Y_test)

print("R^2 Score (Test):",Score)

joblib.dump(Model_fit, r"D:\AI and ML lrn\Week 1\House prediction model\models\linear_regression_model.pkl")
