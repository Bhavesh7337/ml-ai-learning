import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



df = pd.read_csv('D:\AI and ML lrn\Week 1\House prediction model\Housing.csv')

#print(df.head())

#print(df.info())
#print(df.describe())    


x1 = df['price']
y1 = df['area']
y2 = df['bedrooms']
y3 = df['bathrooms']
y4 = df['stories']
y5 = df['parking']
y6 = df['mainroad']
y7 = df['guestroom']
y8 = df['basement']
y9 = df['hotwaterheating']
y10 = df['airconditioning']
y11 = df['prefarea']

histogram_plot = plt.hist(x1, bins=25, color='blue', alpha=0.7)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\price_distribution_histogram.png")


scatter_plot = plt.scatter(y1, x1, color='green', alpha=0.5)
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\area_vs_price_scatter.png")

scatter_plot2 = plt.scatter(y2, x1, color='green', alpha=0.5)
plt.title('Bedroom vs Price')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\bedrooms_vs_price_scatter.png")



boxplot = plt.boxplot(x1, vert=True, patch_artist=True)
plt.title('Price Boxplot')
plt.xlabel('Price')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\price_boxplot.png")

df["mainroad"] = df["mainroad"].map({"yes":1, "no":0})
df["guestroom"] = df["guestroom"].map({"yes":1, "no":0})
df["basement"] = df["basement"].map({"yes":1, "no":0})
df["hotwaterheating"] = df["hotwaterheating"].map({"yes":1, "no":0})
df["airconditioning"] = df["airconditioning"].map({"yes":1, "no":0})
df["prefarea"] = df["prefarea"].map({"yes":1, "no":0})
df["furnishingstatus"] = df["furnishingstatus"].map({"furnished":2, "semi-furnished":1, "unfurnished":0})
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.savefig(r"D:\AI and ML lrn\Week 1\House prediction model\reports\correlation_matrix_heatmap.png")

price_corr = corr_matrix['price'].sort_values(ascending=False)

top_features = price_corr.index[1:6]
print("Top 5 features most correlated with price:" + str(top_features) )

print(df.isnull().sum())

df['mainroad'] = df['mainroad'].replace({'yes': 1, 'no': 2})
df['guestroom'] = df['guestroom'].replace({'yes': 1, 'no': 2})
df['basement'] = df['basement'].replace({'yes': 1, 'no': 2})
df['hotwaterheating'] = df['hotwaterheating'].replace({'yes': 1, 'no': 2})
df['airconditioning'] = df['airconditioning'].replace({'yes': 1, 'no': 2})
df['prefarea'] = df['prefarea'].replace({'yes': 1, 'no': 2})
df['furnishingstatus'] = df['furnishingstatus'].replace({'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1})

X = df.drop(['price'], axis=1)
Y = df['price']


print(X.shape)











