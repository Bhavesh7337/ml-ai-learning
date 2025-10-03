from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib

def preprocess_data(path):
    df = pd.read_csv(path)


    df['mainroad'] = df['mainroad'].replace({'yes': 1, 'no': 0})
    df['guestroom'] = df['guestroom'].replace({'yes': 1, 'no': 0})
    df['basement'] = df['basement'].replace({'yes': 1, 'no': 0})
    df['hotwaterheating'] = df['hotwaterheating'].replace({'yes': 1, 'no': 0})
    df['airconditioning'] = df['airconditioning'].replace({'yes': 1, 'no': 0})
    df['prefarea'] = df['prefarea'].replace({'yes': 1, 'no': 0})
    df['furnishingstatus'] = df['furnishingstatus'].replace({'furnished': 1, 'semi-furnished': 0, 'unfurnished': -1})

    X = df.drop(['price'], axis=1)
    Y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, r'D:\AI and ML lrn\Week 1\House prediction model\models/scaler.pkl')
    np.save(r'D:\AI and ML lrn\Week 1\House prediction model\data/X.npy', X_scaled)
    np.save(r'D:\AI and ML lrn\Week 1\House prediction model\data/y.npy', Y.to_numpy())

    return X_scaled, Y

if __name__ == "__main__":
    path = r"D:\AI and ML lrn\Week 1\House prediction model\Housing.csv"
    X_scaled, Y = preprocess_data(path)
    print("✅ Preprocessing complete. Files saved!")