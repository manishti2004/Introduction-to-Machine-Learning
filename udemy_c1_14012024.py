import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn import linear_model
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/h.csv')
df.head()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Perdiction")
plt.scatter(df["Area"], df["Price"])

Area=df.drop('Price', axis = 'columns')
Area

Price=df.Price
Price

reg =linear_model.LinearRegression()
reg.fit(Area, Price)

reg.predict ([[1200]])
