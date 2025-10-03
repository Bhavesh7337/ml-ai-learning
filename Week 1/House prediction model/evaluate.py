import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve



X_train = np.load(r"D:\AI and ML lrn\Week 1\House prediction model\data\X_train.npy")
Y_train = np.load(r"D:\AI and ML lrn\Week 1\House prediction model\data\Y_train.npy")
X_test = np.load(r"D:\AI and ML lrn\Week 1\House prediction model\data\X_test.npy")
Y_test = np.load(r"D:\AI and ML lrn\Week 1\House prediction model\data\Y_test.npy")

model_use = joblib.load(r"D:\AI and ML lrn\Week 1\House prediction model\models\linear_regression_model.pkl")

model_predictions = model_use.predict(X_test)

mae = mean_absolute_error(Y_test, model_predictions)
mse = mean_squared_error(Y_test, model_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, model_predictions)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)     
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)


scatterplot_Y = plt.scatter(Y_test, model_predictions, color='blue', alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\actual_vs_predicted_scatter.png")

residuals = Y_test - model_predictions
residuals_plot = plt.scatter(model_predictions, residuals, color='red', alpha=0.5)
plt.title('Residuals vs Predicted Prices')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')     
plt.axhline(y=0, color='black', linestyle='--')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\residuals_vs_predicted_scatter.png")

plt.hist(residuals, bins=25, color='green', alpha=0.7)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency') 
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\residuals_distribution_histogram.png") 


learning_curve = learning_curve(model_use, X_train, Y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_sizes, train_scores, test_scores = learning_curve
plt.title('Learning Curve') 
plt.xlabel('Number of Samples')
plt.ylabel('Mean Squared Error')    
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\learning_curve.png")
