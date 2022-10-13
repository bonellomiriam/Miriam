import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import operator 
import pandas as pd


data = pd.read_csv('Q2__HR_Diagram.csv')

data.groupby('Star type')
plt.scatter((data['Luminosity(L/Lo)']),(data['Temperature/K']))
plt.savefig('Plot1.png', dpi=800)

plt.close()

plt.scatter(np.log(data['Luminosity(L/Lo)']), np.log(data['Temperature/K']))
plt.savefig('Plot2.png', dpi=800)


#for i in range 
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
