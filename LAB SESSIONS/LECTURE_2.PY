import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import operator 
import pandas as pd
from math import pi
from scipy.optimize import curve_fit

a_list = []

#to read the data 
data_1 = pd.read_csv('Q2a__HR_Diagram.csv')

#to group the data according to the star type 
data_1.groupby('Star type')

#setting the parameters for the plots 
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

plt.scatter((data['Temperature/K']),(data['Luminosity(L/Lo)']), color = 'k')
plt.savefig('PLOT1.png', dpi=800)

plt.close()

plt.scatter(np.log(data['Temperature/K']), np.log(data['Luminosity(L/Lo)']), color = 'k')
plt.savefig('PLOT2.png', dpi=800)

plt.close()

#to drop any of the unwanted values 
value3 = data.mask(data['Star Type']!=3).dropna().reset_index()
plt.scatter(np.log(value3['Temperature/K']), np.log(value3['Luminosity(L/Lo)']), color = 'k')
plt.savefig('PLOT3.png', dpi=800)

plt.close()

#to get the log of the data needed 
y = np.log(value3['Luminosity(L/Lo)'])
x = np.log(value3['Temperature/K'])

#to reshape the array and list the number of degree values that where tried 
xa = x.array.reshape(-1,1)
freedom = np.array([2, 10, 20, 30, 15, 16, 17])

#a for loop was created to test each degree until the smallest root mean square value was obtained and each test was plotted to see the output 
for i in freedom:
    poly = pf(degree = i)
    poly_lumen = poly.fit_transform(xa)

    model = LinearRegression()
    model.fit(poly_lumen, y)
    y_predicted = model.predict(poly_lumen)

    RMSE = np.sqrt(mean_squared_error(y, y_predicted))
    print(f'The root mean square value is: {RMSE}, the degree of freedom is: {i}')

    plt.scatter(x, y, color='k')
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_predicted), key = sort_axis)
    x, y_predicted = zip(*sorted_zip)
    plt.plot(x, y_predicted, color='k')
    plt.savefig(f'PLOT4_{i}.png', dpi=800)

    plt.close()

#reading the filtered data 
data_2 = pd.read_csv('Q2b__stars.csv')


#defining the variables used in the equation 
temp = data_2['Temperature/K']
lumen = (data_2['Luminosity(L/Lo)'])*(3.846e26)
rad = data_2['Radius(R/Ro)']

#using a for loop to obtain the value of A 
for i in range(len(rad)):
    a = 4 * pi * ((rad[i] * 6.957e8) ** 2)
    a_list.append(a)

A = np.array(a_list)

#to calculate L divided by A
l_a = lumen/A

#creating a function to be able to fit the data 
def fit_func(temp, sigma):
    return sigma * (np.power(temp, 4))

#using linspace so as to have a smooth curve and using curve fit to be  able to produce the best fitting curve 
temp_lin = np.linspace(temp.min(), temp.max(), 1000)
popt, pcov = curve_fit(fit_func, temp, l_a)
fit_line = fit_func(temp_lin, popt[0])

#to plot and obtain the graph 
f = plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.title(r'A graph of $\frac{\mathrm{L}}{\mathrm{A}}$ against T')
plt.scatter(temp, l_a, color='k')
plt.plot(temp_lin, fit_line, color='k')
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.xlabel(r'T/K')
plt.ylabel(r'$\frac{\mathrm{L}}{\mathrm{A}}$ /Wm$^{-2}$')
plt.savefig('PLOT5.png', dpi=800)
plt.show()

#to show the Boltzmann constant 
print(f'Boltzmann constant = {popt[0]:.2E}')