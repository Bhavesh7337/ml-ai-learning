import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_iris


x = np.linspace(0,2*np.pi,100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)


plt.plot(x,y1,label='Sine',color='blue')
plt.plot(x,y2,label='Cosine',color='red')
plt.plot(x,y3,label='Tangent',color='green')
plt.legend()
plt.title('Trigonometric Functions')
plt.show()


data = load_iris(as_frame=True)
df = data.frame

sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue="target", data=df)
plt.title("Iris Petal Length vs Width")
plt.show()



