import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def preprocess_data(path):
    df = pd.read_csv(path)

    # Replace 0s with NaN in specific columns
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI',]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    


    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42,stratify=y)

    np.save(r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data/X_train.npy", X_train)
    np.save(r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data/X_test.npy", X_test)
    np.save(r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data/y_train.npy", y_train)
    np.save(r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data/y_test.npy", y_test)  

    print("Preprocessing complete. Data saved to 'data/' directory.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\diabetes.csv"
    )   



