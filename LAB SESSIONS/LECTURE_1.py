import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#empty lists to be used later on 
x_list = []
y_list = []
prod_list = []
denom_list = []
ydenom_list = []
ye_list = []

#to be able to read the data 
data = pd.read_csv('Q1__Youngs_Modulus_of_a_Wire.csv')
#setting the columns into numpy arrays 
diameter = (data['Diameter/m']).to_numpy()
mass = data['m/kg'].to_numpy()
x0 = data['x_0/m'].to_numpy()
x1 = data['x_1/m'].to_numpy()
x2 = data['x_2/m'].to_numpy()
x3 = data['x_3/m'].to_numpy()
x4 = data['x_4/m'].to_numpy()
L = data['L/m'].to_numpy()

#calculating x-x0
for i in range(len(x1)-1):
    x = sum([x1[i], x2[i], x3[i], x4[i]])
    X = (x/4)-x0[0]
    x_list.append(abs(X))

#calculating x_i and x_bar
x_i = np.array(x_list)**2
x_bar = np.mean(x_i)
print(x_bar)

#calculating yi and y bar
for i in range(len(mass)-1):
    y_i = mass[i]/x_list[i]
    y_list.append(y_i)

y_i = np.array(y_list)
y_bar = np.mean(y_i)
print(y_bar)

#creating a function to be able to calculate the alpha function, the beta and the coefficient of correlation along with the determination 
def beta_alpha_function(x_i, x_bar, y_i, y_bar):
    for i in range(len(x_i)):
        prod = (x_i[i]-x_bar)*(y_i[i]-y_bar)
        prod_list.append(prod)
        xdenom = (x_i[i]-x_bar)**2
        denom_list.append(xdenom)
        ydenom = (y_i[i]-y_bar)**2
        ydenom_list.append(ydenom)
    prodarray = np.array(prod_list)
    denomarray = np.array(denom_list)
    ydenomarray = np.array(ydenom_list)
    numerator = sum(prodarray)
    denominator = sum(denomarray)
    ydenominator = sum(ydenomarray)
    beta = numerator/denominator
    alpha = y_bar - (beta*x_bar)
    r = numerator/(np.sqrt(denominator*ydenominator))
    delta_beta = (beta/(np.sqrt(len(x_i)-2)))*(np.sqrt((1/r**2)-1))
    delta_alpha = delta_beta*np.sqrt(((1/len(x_i))*(sum(x_i**2))))
    R = r**2
    return beta, alpha, delta_beta, delta_alpha, r, R

#to obtain the actual values 
beta, alpha, delta_beta, delta_alpha, r, R = beta_alpha_function(x_i, x_bar, y_i, y_bar)

#to obtain a value for the experimental value 
for i in range(len(x_i)):
    ye = alpha + (beta*x_i[i])
    ye_list.append(ye)
ye_array = np.array(ye_list)

#to obtain the concepts for the intercept and gradient 
radius = np.average(diameter)
m_constant = (8*np.pi*(radius**2))/(9.81*(L[0]**3))
c_constant = 4/(L[0]*9.81)

#obtaining the best fit line 
coeffs, cov = np.polyfit(x_i, ye_array, 1, cov=True)
polyfunc = np.poly1d(coeffs)
trendline = polyfunc(x_i)

#obtaing Young's Modulus and T0
E = coeffs[0]/m_constant
T0 = coeffs[1]/c_constant
delta_E = np.sqrt(cov[0][0])
delta_T0 = np.sqrt(cov[1][1])
print(E, delta_E, T0, delta_T0)

#obtaining a value for the residual 
residual = np.subtract(y_i,trendline)

F, (A_0, A_1) = plt.subplots(2, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7.3, 10.7))

#setting the font for the graph and setting the paraments to obtain the graphsa s one subplot 
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

A_0.scatter(x_i, y_i, color='k', label='Data Points')
A_0.plot(x_i, trendline, color='k', label='Trendline')
A_0.minorticks_on()
A_0.grid(visible=True, which='major', linestyle='-')
A_0.grid(visible=True, which='minor', linestyle='--')
A_0.set_xlabel('Strain')
A_0.set_ylabel('Stress')
A_0.set_title('A graph showing Stress against Strain')

A_1.scatter(x_i, residual, color='k')
A_1.minorticks_on()
A_1.grid(visible=True, which='major', linestyle='-')
A_1.grid(visible=True, which='minor', linestyle='--')
A_1.set_ylabel('Residuals')
A_1.set_xlabel('Strain')
A_1.set_title('A graph showing the Residuals against Strain')
F.tight_layout()
F.savefig('PLOT1.png', dpi=800)
F.show()