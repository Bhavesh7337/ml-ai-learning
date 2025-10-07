"""
run.py — Main script to execute the Diabetes Prediction ML pipeline
Author: Bhavesh Jakhete
"""

# === Imports ===
import os
import numpy as np
from src import train, eda, preprocessing

# === Define Paths ===
DATA_DIR = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data"
MODEL_DIR = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\models"
REPORT_DIR = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\reports"

def main():
    print(" Starting Diabetes Prediction Pipeline...\n")


    print(" Loading preprocessed data...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
        print(" Data loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        return

    print(" Starting model training and evaluation...")
    try:
        train.train_and_evaluate(X_train, X_test, y_train, y_test)
        print(" Training and evaluation completed successfully.\n")
    except Exception as e:
        print(f" Error during training: {e}")
        return

    print(" Pipeline finished successfully! Models and reports are saved in:")
    print(f"Models: {MODEL_DIR}")
    print(f"Reports: {REPORT_DIR}\n")


if __name__ == "__main__":
    main()
