import numpy as np
import matplotlib.pyplot as plt

l_out = np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_1_last.dat")
c_out = np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_2_last.dat")
r_out = np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Left_bp_mon_3_last.dat")

l_out = np.transpose(l_out)
c_out = np.transpose(c_out)
r_out = np.transpose(r_out)

wl = l_out[0]

# working out T and C coeff: note that input flux = 1 always
t_coeff = l_out[1]**0.5
c_coeff_c = c_out[1]**0.5
c_coeff_r = r_out[1]**0.5

# print(t_coeff)

# # 1. Plotting T and C coeffs with wavelength
# plt.figure(1)
# plt.plot(wl, t_coeff, linestyle='dashdot', label='T coeff')
# plt.plot(wl, c_coeff_c, '+', label='C coeff (centre)')
# plt.plot(wl, c_coeff_r, linestyle='dashdot', label='C coeff (right)')
# plt.plot(wl, t_coeff**2 + c_coeff_r**2 + c_coeff_c**2, label='Total (sq)')
# plt.xlabel('Wavelength (micron)')
# plt.ylabel('Intensity (normalised)')
# plt.legend(loc='best')
# plt.ylim(0,1.1)
# plt.grid(True)

# # # 2. Plotting the relation from Fang et al. 
# t_test = np.sqrt(1 - c_coeff_c**2 - c_coeff_r**2)

# plt.figure(2)
# plt.plot(wl, t_coeff, linestyle='dashdot', label='Experimental T coeff')
# plt.plot(wl, t_test, linestyle='dashdot', label='Calculated T coeff')
# plt.grid(True)
# plt.xlabel('Wavelength (micron)')
# plt.ylabel('Intensity (normalised)')
# plt.legend(loc='best')

# 2.5 Plotting the difference to see better:
# plt.figure(3)
# diff = abs(t_coeff - t_test)
# plt.plot(wl, diff, linestyle='dashdot', label='Difference')
# plt.grid(True)
# plt.xlabel('Wavelength (micron)')
# plt.ylabel('Intensity (normalised)')
# plt.legend(loc='best')
# plt.ylim(0,0.02)


# # 2. Testing phi
# plt.figure(4)
# phi_c = np.arccos(-c_coeff_c / (2*t_coeff))
# phi_r = np.arccos(-c_coeff_r / (2*t_coeff))
# phi_avg = np.mean(np.array([phi_c, phi_r]), axis=0)
# plt.plot(wl, phi_c, linestyle='dashdot', label='phi_c')
# plt.plot(wl, phi_r, linestyle='dashdot', label='phi_r')
# plt.plot(wl, phi_avg, label='Average')
# plt.xlabel('Wavelength (micron)')
# plt.ylabel('Angle (rad)')
# plt.legend(loc='best')
# plt.grid(True)

# 3. What if... we found an average C coeff value? 
c_avg = np.mean(np.array([c_coeff_c, c_coeff_r]), axis=0)

plt.figure(5)
plt.plot(wl, t_coeff, linestyle='dashdot', label='T coeff')
plt.plot(wl, c_avg, linestyle='dashdot', label='C coeff avg')
plt.plot(wl, t_coeff**2 + 2*c_avg**2, label='Total')
plt.xlabel('Wavelength (micron)')
plt.ylabel('Intensity (normalised)')
plt.legend(loc='best')
plt.grid(True)

