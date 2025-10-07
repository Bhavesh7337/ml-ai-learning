import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reports = r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\reports"


def basic_checks(path):
    df = pd.read_csv(path)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nDataset Shape:")
    print(df.shape)


    # Check for missing values
    print("\nMissing Values in Each Column:")
    print(df.isnull().sum())    
    print('BMI:' + str((df['BMI'] == 0).sum()))
    print('Glucose:' + str((df['Glucose'] == 0).sum()))
    print('BloodPressure:' + str((df['BloodPressure'] == 0).sum()))
    print('SkinThickness:' + str((df['SkinThickness'] == 0).sum()))
    print('Insulin:' + str((df['Insulin'] == 0).sum()))
    print('Age:' + str((df['Age'] == 0).sum()))
    print('DiabetesPedigreeFunction:' + str((df['DiabetesPedigreeFunction'] == 0).sum()))
    print('Pregnancies:' + str((df['Pregnancies'] == 0).sum()))
    print('Outcome:' + str((df['Outcome'] == 0).sum()))

    # Plot distributions of key features
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(f"{reports}/age_distribution.png") 
    plt.close()  
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['BMI'], bins=30, kde=True)
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.savefig(f"{reports}/bmi_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(f"{reports}/correlation_heatmap.png")
    plt.close()

    #BMI vs Outcome
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y='BMI', data=df)
    plt.title('BMI vs Diabetes Outcome')
    plt.xlabel('Diabetes Outcome')
    plt.ylabel('BMI')
    plt.savefig(f"{reports}/bmi_vs_outcome.png")
    plt.close()

if __name__ == "__main__":
    basic_checks(r"D:\AI and ML lrn\Week 2\DiabetesPredictionModel\data\diabetes.csv")    




