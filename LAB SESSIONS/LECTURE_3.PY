from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

#creating the lists to be used later on
index = []
slice = []
L = []
half_life = []
Z = []
N = []
new = []
new_Z = []
new_N = []

#to read the data 
data_1 = pd.read_csv('Q3__Isotope_Decay_Dataset.csv')

#defining the array
even_A = np.arange(0, 5922, 2)
odd_A = np.arange(1, 5921, 2)

#importing the data and creating an index with the selected data 
A_values = data_1['A']
index_arr = np.arange(0, len(A_values)+20, 20)

#to obtain the mean values of A of each isotope as well as obtaining the 20 values of each isotope
mean_A = [np.mean(A_values.iloc[index_arr[i]:index_arr[i+1]]) for i in range(len(index_arr)-1)]

#to see which values are lower than 95
for i in range(len(mean_A)):
    if mean_A[i] < 95:
        index.append(i)

#creating a new list for the data to sort according to the indices and keeping only the data of the unstable isotopes 
for i in range(len(index)):
    slice.append((index[i]) * 20)
    slice.append((index[i]+1) * 20)
sliced_data = [data_1.iloc[slice[even_A[i]]:slice[odd_A[i]]] for i in range(len(even_A)-1)]

#getting rid of the empty values as well as defining en empty data frame to be used later on
sliced_data = list(filter(lambda df: not df.empty, sliced_data))
df = pd.DataFrame(columns=['z', 'n', 't/s', 'A'])

for i in range(len(sliced_data)):
    temp_df = pd.DataFrame(sliced_data[i], columns = ['z', 'n', 't/s', 'A'])
    df = pd.concat([df, temp_df]).reset_index(drop=True)

#creating an index of the new data frame and selecting the data for A and t only
index_df = np.arange(0, len(df), 1)
df.reset_index(drop=True).set_index(index_df, inplace=True)

A_uvalues = df['A']
T_values = df['t/s']

#to find the mean values for A and t of each isotope 
mean_UA = [np.mean(A_uvalues.iloc[index[i]:index[i+1]]) for i in range(len(index)-1)]
mean_T = [np.mean(T_values.iloc[index[i]:index[i+1]]) for i in range(len(index)-1)]

#setting the parameters to plot the graphs
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()
plt.scatter(np.log(mean_T), np.log(mean_UA), color='k')
plt.savefig('PLOT1_3.png', dpi=800)


#selecting the specific data for calcium and for 1 calcium isotope and then re indexing the data 
calcium_df = df[df['z']==20]
calcium_df = df.iloc[5680:6080]
calcium_df.reset_index(drop=True, inplace=True)

#setting the values and data used for calcium and defining the function to find the value of A/A0
calcium_A = calcium_df['A'][0:20]
calcium_log_A = np.log(calcium_df['A'][0:20])
calcium_T = (calcium_df['t/s'][0:20])

def fit_func(t, t_half):
    return(np.exp((-1 * t * np.log(2)) / t_half))

#making use of curve fit to be able to calculate the value of the half life and then obtaining the curve of the calcium isotope decay 
popt, pcov = curve_fit(fit_func, calcium_T, (calcium_A / calcium_A[0]))
fitted_line = fit_func(calcium_T, popt[0])
print(f'The half-life of Calcium-14 is said to be: {popt[0]:.2E}s')

#to obtain a straight line graph 
coeffs, cov = np.polyfit(calcium_T, calcium_log_A, 1, cov=True)
polyfunc = np.poly1d(coeffs)
trendline = polyfunc(calcium_T)
print(f'The value of the half-life of Calcium-14 obtained from the straight line graph was found to be: {-np.log(2)/coeffs[0]:.2E}s')

#setting the parameters for the plots and subplots to be plotted 
F, (A_0, A_1) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(7.3, 10.7))

A_0.scatter(calcium_T, (calcium_A/calcium_A[0]), color='k', label='Data Points')
A_0.plot(calcium_T, fitted_line, color='k', label='Trendline')
A_0.minorticks_on()
A_0.grid(visible=True, which='major', linestyle='-')
A_0.grid(visible=True, which='minor', linestyle='--')
A_0.set_xlabel('Time/s')
A_0.set_ylabel('A/A0')
A_0.set_title(r'A graph $\frac{A}{A_0}$ vs Time')

A_1.scatter(calcium_T, calcium_log_A, color='k', label='Data Point')
A_1.plot(calcium_T, trendline, color='k', label='Trendline')
A_1.minorticks_on()
A_1.grid(visible=True, which='major', linestyle='-')
A_1.grid(visible=True, which='minor', linestyle='--')
A_1.set_xlabel('Time/s')
A_1.set_ylabel(r'$log(A)$')
A_1.set_title(r'A graph of $log(A)$ vs Time')

F.tight_layout()
F.legend()
F.savefig('PLOT2_3.png', dpi=800)


data_2_T = df['t/s']
data_2_A = df['A']

selection = np.arange(0, len(df['t/s']), 20)

#making use of the curve fit function across the data 
for i in range(len(selection)-1):
    data_A = data_A[selection[i]:selection[i+1]]
    data_T = data_2_T[selection[i]:selection[i+1]]
    data_0 = data_A[selection[i]]
    data__0 = data_A / data_0
    popt, pcov = curve_fit(fit_func, data_T, data__0)
    half_life.append(popt[0])
    Z.append(df['z'][selection[i]])
    N.append(df['n'][selection[i]])

#combining all 3 different lists together and creating a data frame for the values of z, n and the half life 
plotting = list(zip(Z, N, half_life))
results = pd.DataFrame({'Z':Z, 'N':N, 'Half Life/s':half_life})
plot = results.pivot(index = 'Z', columns = 'N', values = 'Half Life/s')

heat_map = sns.heatmap(plot, square=True, norm=LogNorm())
heat_map.invert_yaxis()

#indexing for when Z=N and selecting the data for when Z=N
for i in range(len(results)):
    if results['Z'][i] == results['N'][i]:
        new.append(i)
for i in new:
    new_Z.append(Z[i])
    new_N.append(N[i])

Z_N = pd.DataFrame({'Z': new_Z, 'N':new_N})

#to plot the data and save the plot produced 
plt.plot(Z_N['N'], Z_N['Z'], color='k')
plt.savefig('PLOT3_3.png', dpi=800)