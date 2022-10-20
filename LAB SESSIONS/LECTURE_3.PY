import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import preprocessing 
from scipy.optimize import curve_fit


data = pd.read_csv('Q3__Isotope_Decay_Dataset.csv')
z_array = (data['z']).to_numpy()
n_array = (data['n']).to_numpy()
t_array = (data['t/s']).to_numpy()
A_array = (data['A']).to_numpy()

plt.step(A_array,t_array, 'cs', where='pre')
plt.show()
#pd.set_option('display.max.rows', None)



#print(data)
