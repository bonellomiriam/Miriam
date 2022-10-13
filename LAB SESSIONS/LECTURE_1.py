import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# creating the empty lists to be used
xlist = []
ylist = []
prodlist = []
denomlist = []
ydenomlist = []
yelist = []

# reading the data
data = pd.read_csv('Q1__Youngs_Modulus_of_a_Wire.csv')
# changing the individual columns into arrays
diameter_array = (data['Diameter/m']).to_numpy()
mass_array = data['m/kg'].to_numpy()
x1_array = data['x_1/m'].to_numpy()
x2_array = data['x_2/m'].to_numpy()
x3_array = data['x_3/m'].to_numpy()
x4_array = data['x_4/m'].to_numpy()
L_array = data['L/m'].to_numpy()
x0_array = data['x_0/m'].to_numpy()

# finding the x-x0 part of the equation
for i in range(len(x1_array)-1):
    x = sum([x1_array[i], x2_array[i], x3_array[i], x4_array[i]])
    X = (x/4)-x0_array[0]
    xlist.append(abs(X))

# calculating xi and xbar
xi = np.array(xlist)**2
xbar = np.mean(xi)
print(xbar)

# calculating yi and y bar
for i in range(len(mass_array)-1):
    yi = mass_array[i]/xlist[i]
    ylist.append(yi)

yi = np.array(ylist)
ybar = np.mean(yi)
print(ybar)

# defining the function to calculate alpha, beta, coefficient of correlation and determination
def beta_alpha_function(xi, xbar, yi, ybar):
    for i in range(len(xi)):
        prod = (xi[i]-xbar)*(yi[i]-ybar)
        prodlist.append(prod)
        xdenom = (xi[i]-xbar)**2
        denomlist.append(xdenom)
        ydenom = (yi[i]-ybar)**2
        ydenomlist.append(ydenom)
    prod_array = np.array(prodlist)
    denom_array = np.array(denomlist)
    ydenom_array = np.array(ydenomlist)
    numerator = sum(prod_array)
    denominator = sum(denom_array)
    ydenominator = sum(ydenom_array)
    beta = numerator/denominator
    alpha = ybar - (beta*xbar)
    r = numerator/(np.sqrt(denominator*ydenominator))
    delta_beta = (beta/(np.sqrt(len(xi)-2)))*(np.sqrt((1/r**2)-1))
    delta_alpha = delta_beta*np.sqrt(((1/len(xi))*(sum(xi**2))))
    R = r**2
    return beta, alpha, delta_beta, delta_alpha, r, R

# calling previous function to get the actual values
beta, alpha, delta_beta, delta_alpha, r, R = beta_alpha_function(xi, xbar, yi, ybar)

# calculating the experimental values 
for i in range(len(xi)):
    ye = alpha + (beta*xi[i])
    yelist.append(ye)
ye_array = np.array(yelist)

# calculating the constants for the gradient and intercept
radius = np.average(diameter_array)
m_constant = (8*np.pi*(radius**2))/(9.81*(L_array[0]**3))
c_constant = 4/(L_array[0]*9.81)

# finding the line of best fit to the data
coeffs, cov = np.polyfit(xi, ye_array, 1, cov=True)
polyfunc = np.poly1d(coeffs)
trendline = polyfunc(xi)

# calculating the young's modulus and T0
E = coeffs[0]/m_constant
T0 = coeffs[1]/c_constant
delta_E = np.sqrt(cov[0][0])
delta_T0 = np.sqrt(cov[1][1])
print(E, delta_E, T0, delta_T0)

# calculating the residuals
residual = np.subtract(yi,trendline)

f, (a0, a1) = plt.subplots(2, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [3, 1]}, figsize=(7.3, 10.7))

# defining the font to be used
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

# plotting both graph as subplots
a0.scatter(xi, yi, color='k', label='Data Points')
a0.plot(xi, trendline, color='k', label='Trendline')
a0.minorticks_on()
a0.grid(visible=True, which='major', linestyle='-')
a0.grid(visible=True, which='minor', linestyle='--')
a0.set_xlabel('Strain')
a0.set_ylabel('Stress')
a0.set_title('A graph of Stress vs Strain')

a1.scatter(xi, residual, color='k')
a1.minorticks_on()
a1.grid(visible=True, which='major', linestyle='-')
a1.grid(visible=True, which='minor', linestyle='--')
a1.set_ylabel('Residuals')
a1.set_xlabel('Strain')
a1.set_title('A graph of Residuals vs Strain')
f.tight_layout()
f.savefig('Plot1.png', dpi=800)
f.show()
