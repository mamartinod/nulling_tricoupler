import h5py
import numpy as np

zeta_file = h5py.File("20210322_zeta_coeff_raw.hdf5", 'r')

# print(np.array(zeta_file))

wl_scale = np.array(zeta_file['wl_scale'])

# null output for baseline 1, beam 1
zeta_b1_n1 = np.array(zeta_file['b1null1']) # Conversion in numpy array is mandatory to do calculations with the values
zeta_b1_an1 = np.array(zeta_file['b1null7'])
zeta_b1_n3 = np.array(zeta_file['b1null3'])
zeta_b1_an3 = np.array(zeta_file['b1null9'])
zeta_b1_n5 = np.array(zeta_file['b1null5'])
zeta_b1_an5 = np.array(zeta_file['b1null11'])

alpha_b1 = (zeta_b1_n1 + zeta_b1_an1) / (1 + (zeta_b1_n1 + zeta_b1_an1) + (zeta_b1_n3 + zeta_b1_an3) + (zeta_b1_n5 + zeta_b1_an5))

zeta_b2_n1 = np.array(zeta_file['b2null1']) # Conversion in numpy array is mandatory to do calculations with the values
zeta_b2_an1 = np.array(zeta_file['b2null7'])
zeta_b2_n2 = np.array(zeta_file['b2null2'])
zeta_b2_an2 = np.array(zeta_file['b2null8'])
zeta_b2_n6 = np.array(zeta_file['b2null6'])
zeta_b2_an6 = np.array(zeta_file['b2null12'])





# formerly named gamma
alpha_b2 = (zeta_b2_n1 + zeta_b2_an1) / (1 + (zeta_b2_n1 + zeta_b2_an1) + (zeta_b2_n2 + zeta_b2_an2) + (zeta_b2_n6 + zeta_b2_an6))

print(alpha_b2)









# null output for baseline 1, beam 2
# zeta_b2_n1 = np.array(zeta_file['b2null1'])
#
# # antinull output for baseline 1, beam 2
# zeta_b2_an1 = np.array(zeta_file['b2null7'])
#
# kappa_12 = (zeta_b1_an1 / zeta_b1_n1) / (1 + (zeta_b1_an1 / zeta_b1_n1))
#
# kappa_21 = (zeta_b2_n1 / zeta_b2_an1) / (1 + (zeta_b2_n1 / zeta_b2_an1))
#
# # alpha = () / (1 + () + () + ())
# print(zeta_file.keys())


# alpha = (zeta_b1_an1 / kappa_12) / (1 + (zeta_b1_an1 / kappa_12))
# gamma = (zeta_b2_n1 / kappa_21) / (1 + (zeta_b2_n1 / kappa_21))


# import matplotlib.pyplot as plt
# #   test: seeing if the wavelength intervals are evenly spaced apart
# plt.figure(figsize=(14,4))
# plt.scatter(wl_scale, np.ones(len(wl_scale)), marker='+')
# plt.show()


print("worked")

zeta_file.close()
