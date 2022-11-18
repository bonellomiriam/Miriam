import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

data_1 = pd.read_csv('Q4__Galilean_Moon_Astrometric_Data.csv')

#to import the data of each moon 
IO_pos = data_1['Io_Offset (Jup Diameters)']
EUROPA_pos = data_1['Europa_Offset (Jup Diameters)']
GANYMEDE_pos = data_1['Ganymede_Offset (Jup Diameters)']
CALLISTO_pos = data_1['Callisto_Offset (Jup Diameters)']

IO_hjd = data_1['Io_Julian_Date (HJD)']
EUROPA_hjd = data_1['Europa_Julian_Date (HJD)']
GANYMEDE_hjd = data_1['Ganymede_Julian_Date (HJD)']
CALLISTO_hjd = data_1['Callisto_Julian_Date (HJD)']

#making use of linspace so as to smoothen the curve of the data 
IO_lin = np.linspace(IO_hjd.min(), IO_hjd.max(), 1000)
EUROPA_lin = np.linspace(EUROPA_hjd.min(), EUROPA_hjd.max(), 1000)
GANYMEDE_lin = np.linspace(GANYMEDE_hjd.min(), GANYMEDE_hjd.max(), 1000)
CALLISTO_lin = np.linspace(CALLISTO_hjd.min(), CALLISTO_hjd.max(), 1000)

#creating a wave function to then be able to plot the data 
def wave_function(t, A, T):
    return A * np.sin(((2 * np.pi) / T) * t)

#creating the curve fit 
popt_IO, pcov_IO = curve_fit(wave_function, IO_hjd, IO_pos, p0 = (max(IO_pos), 1.75))
fitted_IO = wave_function(IO_lin, popt_IO[0], popt_IO[1])

popt_EUROPA, pcov_EUROPA = curve_fit(wave_function, EUROPA_hjd, EUROPA_pos, p0 = (max(EUROPA_pos), 3.56))
fitted_EUROPA = wave_function(EUROPA_lin, popt_EUROPA[0], popt_EUROPA[1])

popt_GANYMEDE, pcov_GANYMEDE = curve_fit(wave_function, GANYMEDE_hjd, GANYMEDE_pos, p0=(max(GANYMEDE_pos), 7.15))
fitted_ganymede = wave_function(GANYMEDE_lin, popt_GANYMEDE[0], popt_GANYMEDE[1])

popt_CALLISTO, pcov_CALLISTO = curve_fit(wave_function, CALLISTO_hjd, CALLISTO_pos, p0=(max(CALLISTO_pos), 16.5))
fitted_callisto = wave_function(CALLISTO_lin, popt_CALLISTO[0], popt_CALLISTO[1])

#defining the subplots and preparing the parameters for the plot  
F, (A_0, A_1, A_2, A_3) = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(7.3, 10.7))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

A_0.minorticks_on()
A_0.grid(visible=True, which='major', linestyle='-')
A_0.grid(visible=True, which='minor', linestyle='--')
A_0.set_xlabel('Julian Date')
A_0.set_ylabel('Offset in Jupiter Diameter')
A_0.scatter(IO_hjd, IO_pos, color='k', label='Io')
A_0.plot(IO_lin, fitted_IO, '--', color='k', label='Io Fit')
A_0.legend()

A_1.minorticks_on()
A_1.grid(visible=True, which='major', linestyle='-')
A_1.grid(visible=True, which='minor', linestyle='--')
A_1.set_xlabel('Julian Date')
A_1.set_ylabel('Offset in Jupiter Diameter')
A_1.scatter(EUROPA_hjd, EUROPA_pos, marker='s', color='k', label='Europa') # type:ignore
A_1.plot(EUROPA_lin, fitted_EUROPA, '--', color='k', label='Europa Fit')
A_1.legend()

A_2.minorticks_on()
A_2.grid(visible=True, which='major', linestyle='-')
A_2.grid(visible=True, which='minor', linestyle='--')
A_2.set_xlabel('Julian Date')
A_2.set_ylabel('Offset in Jupiter Diameter')
A_2.scatter(GANYMEDE_hjd, GANYMEDE_pos, color='k', label='Ganymede')
A_2.plot(GANYMEDE_lin, fitted_ganymede, '--', color='k', label='Ganymede Fit')
A_2.legend()

A_3.minorticks_on()
A_3.grid(visible=True, which='major', linestyle='-')
A_3.grid(visible=True, which='minor', linestyle='--')
A_3.set_xlabel('Julian Date')
A_3.set_ylabel('Offset in Jupiter Diameter')
A_3.scatter(CALLISTO_hjd, CALLISTO_pos, marker='^', color='k', label='Callisto')  # type:ignore
A_3.plot(CALLISTO_lin, fitted_callisto, '--', color='k', label='Callisto Fit')
A_3.legend()

F.tight_layout()
F.savefig('PLOT1_4.png', dpi=800)
plt.show()

plt.close()

# calculating the semi-major axis of each moon in meters
IO_rad = abs(popt_IO[0])*138920000
EUROPA_rad = abs(popt_EUROPA[0])*138920000
GANYMEDE_rad = abs(popt_GANYMEDE[0])*138920000
CALLISTO_rad = abs(popt_CALLISTO[0])*138920000

# displaying the semi-major axis of each moon
print(f'Io semi-major axis is: {IO_rad:.2}m, Europa semi-major axis is: {EUROPA_rad:.2}m, Ganymede semi-major axis is: {GANYMEDE_rad:.2}m, Callisto semi-major axis is: {CALLISTO_rad:.2}m')

# calculating the periodic time of each moon in seconds
IO_period = abs(popt_IO[1])*86400
EUROPA_period = abs(popt_EUROPA[1])*86400
GANYMEDE_period = abs(popt_GANYMEDE[1])*86400
CALLISTO_period = abs(popt_CALLISTO[1])*86400

# displaying the periodic time for each moon
print(f'Io period is: {IO_period:.2f}s, Europa period is: {EUROPA_period:.2f}s, Ganymede period is: {GANYMEDE_period:.2f}s, Callisto period is: {CALLISTO_period:.2f}s')

# defining arrays for radii and periods
radius = np.array([IO_rad, EUROPA_rad, GANYMEDE_rad, CALLISTO_rad])
period = np.array([IO_period, EUROPA_period, GANYMEDE_period, CALLISTO_period])

# finding r^3 and T^2
Y = radius**3
X = period**2

# determining the line of best fit for the given data
coeffs, cov = np.polyfit(X, Y, 1, cov=True)
poly_function = np.poly1d(coeffs)
fit_line = poly_function(X)

# plotting the straight line graph
plt.figure(figsize=(7.5,10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.scatter(X, Y, color='k')
plt.plot(X, fit_line, '-', color='k')
plt.ylabel(r'r$^3$/m$^3$')
plt.xlabel(r'T$^2$/s$^2$')
plt.title(r'A Graph of r$^3$ vs T$^2$')
plt.savefig('PLOT2_4.png', dpi=800)
plt.show()

# determining the gradient of the straight line and the gradient error
gradient = coeffs[0]
gradient_error = np.sqrt(cov[0][0])
# defining the gravitational constant and real jupiter mass
G = 6.6743e-11
real_mass_jupiter =1.898e27
# finding the mass of jupiter
mass_jupiter = (4*(np.pi**2)*gradient)/G
# finding the error of the mass of jupiter calculation
delta_mass_jupiter = np.sqrt(((4*(np.pi**2))/G)*(gradient_error))
precision_jupiter = abs(((mass_jupiter/real_mass_jupiter)-1)*100)
# displaying the results
print(f'The mass of jupiter was found to be: {mass_jupiter:.2E}kg ± {delta_mass_jupiter:.2E} with a precision of: {precision_jupiter:.2f}%')