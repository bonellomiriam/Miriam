import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression

data_air = pd.read_excel('Data.xlsx', 0)
data_short = pd.read_excel('Data.xlsx', 1)
data_water = pd.read_excel('Data.xlsx', 2)
data_methanol = pd.read_excel('Data.xlsx', 3).dropna()
data_NaCl = pd.read_excel('Data.xlsx', 4).dropna()
e_methanol = pd.read_excel('Data.xlsx', 5).dropna()
e_NaCl = pd.read_excel('Data.xlsx', 6).dropna()

frequency = np.array(data_air['%freq[Hz]'])

air_real_avg = data_air[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_air['Real Avg'] = air_real_avg

air_imaginary_avg = data_air[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_air['Imag Avg'] = air_imaginary_avg

water_real_avg = data_water[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_water['Real Avg'] = water_real_avg

water_imaginary_avg = data_water[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_water['Imag Avg'] = water_imaginary_avg

methanol_real_avg = data_methanol[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_methanol['Real Avg'] = methanol_real_avg

methanol_imaginary_avg = data_methanol[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_methanol['Imag Avg'] = methanol_imaginary_avg

NaCl_real_avg = data_NaCl[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_NaCl['Real Avg'] = NaCl_real_avg

NaCl_imaginary_avg = data_NaCl[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_NaCl['Imag Avg'] = NaCl_imaginary_avg

e_methanol_real_avg = e_methanol[["e'_R1","e'_R2","e'_R3"]].mean(axis=1)
e_methanol['Real Avg'] = e_methanol_real_avg

e_methanol_imaginary_avg = e_methanol[["e''_R1","e''_R2","e''_R3"]].mean(axis=1)
e_methanol['Imag Avg'] = e_methanol_imaginary_avg

e_NaCl_real_avg = e_NaCl[["e'_R1","e'_R2","e'_R3"]].mean(axis=1)
e_NaCl['Real Avg'] = e_NaCl_real_avg

e_NaCl_imaginary_avg = e_NaCl[["e''_R1","e''_R2","e''_R3"]].mean(axis=1)
e_NaCl['Imag Avg'] = e_NaCl_imaginary_avg

short_r_avg = data_short['re:Trc1_S11'].to_numpy()
short_i_avg = data_short['im:Trc1_S11'].to_numpy()

air_r_avg = data_air['Real Avg'].to_numpy()
air_i_avg = data_air['Imag Avg'].to_numpy()

water_r_avg = data_water['Real Avg'].to_numpy()
water_i_avg = data_water['Imag Avg'].to_numpy()

methanol_r_avg = data_methanol['Real Avg'].to_numpy()
methanol_i_avg = data_methanol['Imag Avg'].to_numpy()

NaCl_r_avg = data_NaCl['Real Avg'].to_numpy()
NaCl_i_avg = data_NaCl['Imag Avg'].to_numpy()

e_methanol_r_avg = e_methanol['Real Avg'].to_numpy()
e_methanol_i_avg = e_methanol['Imag Avg'].to_numpy()

e_NaCl_r_avg = e_NaCl['Real Avg'].to_numpy()
e_NaCl_i_avg = e_NaCl['Imag Avg'].to_numpy()

air_complex = air_r_avg - (1j * air_i_avg)
short_complex = short_r_avg - (1j * short_i_avg)
water_complex = water_r_avg - (1j * water_i_avg)
methanol_complex = methanol_r_avg - (1j * methanol_i_avg)
NaCl_complex = NaCl_r_avg - (1j * NaCl_i_avg)
e_methanol_c = e_methanol_r_avg - (1j * e_methanol_i_avg)
e_NaCl_c = e_NaCl_r_avg - (1j * e_NaCl_i_avg)

delta_13 = np.subtract(short_complex, water_complex)
delta_21 = np.subtract(air_complex, short_complex)
delta_23 = np.subtract(air_complex, water_complex)
delta_32 = np.subtract(water_complex, air_complex)
delta_m1_methanol = np.subtract(methanol_complex, short_complex)
delta_m2_methanol = np.subtract(methanol_complex, air_complex)
delta_m3_methanol = np.subtract(methanol_complex, water_complex)
delta_m1_NaCl = np.subtract(NaCl_complex, short_complex)
delta_m2_NaCl = np.subtract(NaCl_complex, air_complex)
delta_m3_NaCl = np.subtract(NaCl_complex, water_complex)

e_methanol = -1 * (((delta_m2_methanol * delta_13) / (delta_m1_methanol * delta_32)) * 80.5) - (((delta_m3_methanol * delta_21) / (delta_m1_methanol * delta_32)) * 1)
e_NaCl = -1 * (((delta_m2_NaCl * delta_13) / (delta_m1_NaCl * delta_32)) * 80.5) - (((delta_m3_NaCl * delta_21) / (delta_m1_NaCl * delta_32)) * 1)

methanol_real = (np.real(e_methanol))

methanol_reshape = np.reshape(methanol_real, (-1,1))
poly = pf(degree = 10)
poly_Lumen=poly.fit_transform(methanol_reshape)
model = LinearRegression()
model.fit(poly_Lumen, frequency)
y_pred = model.predict(poly_Lumen)

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.scatter(frequency, np.real(e_methanol), color='k', label='Calculated')
plt.plot(e_methanol, y_pred)
# plt.scatter(frequency, np.real(e_methanol_c), color='r', label='Given')
plt.title('Complex Permittivity of Methanol')
plt.legend()
plt.show()
# plt.savefig('Plot1.png', dpi=800)
plt.close()

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.scatter(frequency, e_NaCl, color='k', label='Calculated')
plt.scatter(frequency, e_NaCl_c, color='k', label='Given')
plt.title('Complex Permittivity of NaCl')
plt.legend()
plt.show()
# plt.savefig('Plot2.png', dpi=800)